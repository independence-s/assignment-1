"""The domain-independent ReAct loop shared by both agents.

Part 1 completes the generic loop here; the two subclasses in this package
supply only their own tools and tool executors.
"""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import yaml

from dotenv import load_dotenv
from openai import OpenAI

from assignment.env import Environment
from assignment.agent.tools import INVOKE_SKILL_TOOL

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_KEEP_RECENT_STEPS = 1
DEFAULT_COMPACTION_MAX_TOKENS = 1_200
MAX_OBSERVATION_CHARS = 10_000

# TODO(Part 2): Write instructions that make the model produce concise working
# memory for a software agent. The prompt should preserve concrete progress,
# failures, test results, constraints, and next steps without copying raw output.
COMPACTION_SYSTEM_PROMPT = (
    "Write concise, factual working memory for a software agent that is "
    "continuing a long terminal session. Preserve: the original objective and "
    "any hard constraints; for every file touched, its path and the exact "
    "change or the reason it was not changed; commands that were run and their "
    "concrete results; test outcomes; failed approaches and why they failed; "
    "current blockers; and the single next action to take. Do not copy raw "
    "command output or code verbatim: restate only the facts a later step "
    "needs. Stay well under the token budget; the agent will see only this "
    "summary plus the most recent step, so the memory must stand alone."
)


class StepLimitError(Exception):
    """Raised when an agent exhausts its model-call budget."""


def format_tool_output(output: dict[str, Any]) -> str:
    """Format a terminal result as a compact, tagged model observation."""

    elements: list[str] = []
    for key in sorted(output):
        value = output[key]
        if isinstance(value, str) and len(value) > MAX_OBSERVATION_CHARS:
            # Leave room for the elision notice so the formatted value itself,
            # not just its retained source slices, stays below the limit.
            retained_at_each_end = 4_900
            omitted = len(value) - (2 * retained_at_each_end)
            value = (
                f"{value[:retained_at_each_end]}\n"
                f"[{omitted} characters elided; read a narrower range]\n"
                f"{value[-retained_at_each_end:]}"
            )
        elements.append(f"<{key}>{value}</{key}>")
    return "\n".join(elements)


def rough_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate prompt tokens without a provider-specific tokenizer."""

    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized) / 4))


def _transcript_text(messages: list[dict[str, Any]]) -> str:
    """Render messages as readable text for a compaction request.

    Tool-call arguments are unwrapped so the model summarizing the transcript
    sees the commands and content rather than their escaped JSON form.
    """

    lines: list[str] = []
    for message in messages:
        role = message.get("role", "?")
        content = message.get("content")
        if isinstance(content, str) and content:
            lines.append(f"<{role}>{content}</{role}>")
        for call in message.get("tool_calls") or []:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            lines.append(
                f"<tool_call>{function.get('name', '?')}: "
                f"{function.get('arguments', '')}</tool_call>"
            )
    return "\n".join(lines)


class Agent:
    """Base class for a ReAct agent with pluggable tools."""

    def __init__(
        self,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        self.env = environment
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise RuntimeError("OPENAI_MODEL is not set.")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is not set.")
        try:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        except ValueError as exc:
            raise RuntimeError("OPENAI_MAX_RETRIES must be an integer.") from exc
        if max_retries < 0:
            raise RuntimeError("OPENAI_MAX_RETRIES must be non-negative.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )

        self.logs_save_path = logs_save_path
        self.step_limit = step_limit
        self.auto_stop_environment = auto_stop_environment
        if compact_threshold_tokens is not None and compact_threshold_tokens <= 0:
            raise ValueError("compact_threshold_tokens must be positive or None")
        if (
            compaction_keep_recent_steps is not None
            and compaction_keep_recent_steps < 1
        ):
            raise ValueError("compaction_keep_recent_steps must be at least 1")
        if compaction_max_tokens is not None and compaction_max_tokens < 1:
            raise ValueError("compaction_max_tokens must be positive")
        # A None threshold turns compaction off. The other two settings then
        # describe a compaction that never happens, so fall back to the
        # defaults rather than leaving a None for later code to trip over.
        self.compact_threshold_tokens = compact_threshold_tokens
        self.compaction_keep_recent_steps = (
            DEFAULT_COMPACTION_KEEP_RECENT_STEPS
            if compaction_keep_recent_steps is None
            else compaction_keep_recent_steps
        )
        self.compaction_max_tokens = (
            DEFAULT_COMPACTION_MAX_TOKENS
            if compaction_max_tokens is None
            else compaction_max_tokens
        )

        # Each agent supplies its own opening messages: the standing
        # instructions, and the task statement that starts the run.
        self.system_prompt: str = ""
        self.task_prompt: str = ""

        self.api_prompts: list[list[dict[str, Any]]] = []
        self.api_responses: list[dict[str, Any]] = []
        self.compaction_events: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.finished = False
        self.steps_taken = 0

        self.skills_path = Path(skills_path) if skills_path is not None else None
        self.skills: dict[str, dict[str, str]] = (
            self.load_skills(self.skills_path) if self.skills_path is not None else {}
        )

        if self.skills:
            self.tools.append(INVOKE_SKILL_TOOL)

        # The working transcript of the run: every assistant message (with its
        # tool calls) followed by the tool observations that answered them,
        # oldest first. build_prompt() replays this after the system and task
        # messages so the model sees its own prior reasoning, actions, and
        # observations on every request.
        self.history: list[dict[str, Any]] = []

        # Working memory produced by context compaction (Part 2): a short
        # message or two inserted after the task prompt that stands in for the
        # oldest turns that compact_context() removed from self.history.
        self.compaction_summary: list[dict[str, Any]] = []

    def load_skills(self, skills_path: Path) -> dict[str, dict[str, str]]:
        """Load the skill folders exposed to this agent."""

        if not Path(skills_path).is_dir():
            raise ValueError(f"skills_path is not a directory: {skills_path}")

        skills: dict[str, dict[str, str]] = {}
        for child in sorted(Path(skills_path).iterdir()):
            if not child.is_dir():
                continue
            skill_file = child / "SKILL.md"
            if not skill_file.is_file():
                raise ValueError(
                    f"Skill directory {child} has no SKILL.md file."
                )
            raw = skill_file.read_text(encoding="utf-8")
            # Frontmatter: a YAML block between `---` lines at the head of the
            # file, followed by the skill body.
            if not raw.startswith("---\n"):
                raise ValueError(f"{skill_file} is missing YAML frontmatter.")
            end = raw.find("\n---", 4)
            if end == -1:
                raise ValueError(f"{skill_file} has unterminated YAML frontmatter.")
            frontmatter_text = raw[4:end]
            body = raw[end + 4:]
            try:
                frontmatter = yaml.safe_load(frontmatter_text)
            except yaml.YAMLError as exc:
                raise ValueError(f"{skill_file} frontmatter is not valid YAML: {exc}") from exc
            if not isinstance(frontmatter, dict):
                raise ValueError(f"{skill_file} frontmatter must be a YAML mapping.")
            name = frontmatter.get("name")
            description = frontmatter.get("description")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{skill_file} frontmatter is missing a 'name'.")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"{skill_file} frontmatter is missing a 'description'.")
            if name in skills:
                raise ValueError(f"Duplicate skill name: {name}")
            metadata_lines = [f"name: {name}", f"description: {description.strip()}"]
            skills[name] = {
                # Concise catalog entry for the system prompt; the full body is
                # kept out until the skill is invoked (progressive disclosure).
                "metadata": "\n".join(metadata_lines),
                "content": body.lstrip("\n").rstrip(),
            }
        return skills

    def query_language_model(self) -> dict[str, Any]:
        """Send one tool-enabled Chat Completions request and normalize it."""

        messages = self.build_prompt()
        self.api_prompts.append(deepcopy(messages))
        step_number = self.steps_taken + 1
        print(
            f"[agent] step {step_number}/{self.step_limit}: requesting action",
            flush=True,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                reasoning_effort="medium",
                max_completion_tokens=4096,
            )
        except Exception as exc:
            print(
                f"[agent] step {step_number}: model request failed after retries "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            raise
        self.api_responses.append(response.model_dump(mode="json"))
        self.steps_taken += 1
        message = self.process_response(response)
        tool_names = [
            call.get("function", {}).get("name", "unknown")
            for call in message.get("tool_calls", [])
            if isinstance(call, dict)
        ]
        if tool_names:
            print(
                f"[agent] step {step_number}: tool call(s): {', '.join(tool_names)}",
                flush=True,
            )
        else:
            print(
                f"[agent] step {step_number}: response contained no parsed tool call; "
                "the loop should preserve the response and continue",
                flush=True,
            )
        return message

    def process_response(self, response: Any) -> dict[str, Any]:
        """Return relevant parts of the language model's response."""

        return response.choices[0].message.model_dump(exclude_none=True)

    def build_prompt(self) -> list[dict[str, Any]]:
        """Assemble the full message sequence for one language-model request.

        Layout: exactly one system message (standing instructions), one user
        message (the task), any working-memory summaries inserted by context
        compaction, then the raw transcript of prior turns (assistant messages
        with their linked tool observations) in order. Domain-agnostic: the
        subclasses supply only ``system_prompt`` and ``task_prompt``.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.task_prompt},
        ]
        messages.extend(deepcopy(self.compaction_summary))
        messages.extend(deepcopy(self.history))
        return messages

    def estimate_active_prompt_tokens(self) -> int:
        """Estimate the next prompt, calibrated by the provider's latest usage."""

        current_prompt = self.build_prompt()
        rough_current = rough_message_tokens(current_prompt)
        if not self.api_prompts or not self.api_responses:
            return rough_current

        usage = self.api_responses[-1].get("usage") or {}
        actual_previous = usage.get("prompt_tokens")
        if not isinstance(actual_previous, int):
            return rough_current

        rough_previous = rough_message_tokens(self.api_prompts[-1])
        added_since_previous_request = max(0, rough_current - rough_previous)
        return actual_previous + added_since_previous_request

    @property
    def compaction_enabled(self) -> bool:
        """Whether this agent compacts its context at all."""

        return self.compact_threshold_tokens is not None

    def compact_context(self):
        """Replace parts of prompt with model-generated working memory. Changes the
        content that `build_prompt` emits."""

        # Everything before the most recent complete steps is fair game for
        # summarization; those steps stay verbatim so the model keeps a valid
        # recent action/observation pair to build on.
        assistant_indices = [
            i for i, m in enumerate(self.history) if m.get("role") == "assistant"
        ]
        keep = self.compaction_keep_recent_steps
        cutoff = assistant_indices[-keep] if len(assistant_indices) > keep else 0
        prefix = self.history[:cutoff]
        #convert to text
        transcript = _transcript_text(prefix)
        compaction_prompt = [
            {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Compact the session below into working memory. The system "
                    "message of the original session and the task statement are "
                    "retained separately and do not need summarizing.\n\n"
                    f"Task:\n{self.task_prompt}\n\n"
                    "Transcript of the older steps to summarize:\n"
                    f"{transcript}"
                ),
            },
        ]

        ### Do not modify this section ###
        compaction_response = self.client.chat.completions.create(
            model=self.model,
            messages=compaction_prompt,
            reasoning_effort="medium",
            max_completion_tokens=self.compaction_max_tokens,
        )
        ##################################

        # Drop the summarized prefix from the working transcript and expose the
        # model-generated summary through build_prompt(). api_prompts and
        # api_responses are untouched: they stay the auditable log of requests.
        summary = compaction_response.choices[0].message.content
        if not summary or not summary.strip():
            summary = "(The model returned an empty working-memory summary.)"
        self.compaction_summary = [
            {
                "role": "user",
                "content": (
                    "Earlier steps were compacted into this working memory; treat "
                    "it as the current state of the session:\n\n"
                    f"{summary.strip()}"
                ),
            }
        ]
        if cutoff:
            self.history = self.history[cutoff:]

        # Use `compaction_response` to update what `build_prompt` emits, but
        # DO NOT modify the object itself. Let the method return it unchanged.

        ### Do not modify this section ###
        return compaction_prompt, compaction_response.model_dump(mode="json")
        ##################################

    def maybe_compact_context(self) -> bool:
        """Compact before the next action request when the threshold is reached."""

        if not self.compaction_enabled:
            return False

        # Context too short to compact yet
        if self.estimate_active_prompt_tokens() < self.compact_threshold_tokens:
            return False

        prompt_before = deepcopy(self.build_prompt())

        # Not enough steps (each assistant turn corresponds to a step) to force
        # compaction yet
        if (
            len([m for m in prompt_before if m.get("role") == "assistant"])
            <= self.compaction_keep_recent_steps
        ):
            return False

        compaction_prompt, compaction_response = self.compact_context()
        prompt_after = deepcopy(self.build_prompt())
        self.compaction_events.append(
            {
                "step": self.steps_taken,
                "estimated_tokens_before": rough_message_tokens(prompt_before),
                "estimated_tokens_after": rough_message_tokens(prompt_after),
                "active_prompt_before": deepcopy(prompt_before),
                "compaction_prompt": compaction_prompt,
                "compaction_response": compaction_response,
            }
        )
        return True

    def run(self) -> None:
        """Run ReAct steps, always saving the trajectory and stopping Modal."""

        try:
            while not self.finished:
                if self.steps_taken >= self.step_limit:
                    raise StepLimitError(
                        "Agent exceeded step limit of "
                        f"{self.step_limit} model calls."
                    )

                # TODO(2.2) Call `maybe_compact_context()` before each new action
                # request in your shared loop. It already estimates active tokens
                # and handles the threshold, and tracks compaction events for
                # logging.
                self.maybe_compact_context()

                # One ReAct iteration: ask the model for the next action, keep
                # its message in the transcript, then execute every tool call it
                # made and append the linked observations. The loop ends when a
                # subclass marks the run finished (a terminal chess state, a
                # user-bound message) or the model answers in plain text.
                message = self.query_language_model()
                self.history.append(message)
                if message.get("tool_calls"):
                    observations = self.execute_tool_calls(
                        message["tool_calls"]
                    )
                    self.history.extend(observations)
                else:
                    # Text-only response: no action to run, so the agent is
                    # done reasoning and the run is complete.
                    self.finished = True
        finally:
            # This block is provided infrastructure. Do not modify it: a
            # trajectory is required even when a run fails.
            if self.logs_save_path:
                path = Path(self.logs_save_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "prompts": self.api_prompts,
                            "responses": self.api_responses,
                            "compactions": self.compaction_events,
                        },
                        indent=2,
                    )
                )
            if self.auto_stop_environment:
                stop = getattr(self.env, "stop", None)
                if callable(stop):
                    stop()

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute domain-specific calls and return linked tool observations."""

        # You do not need to implement anything here. This method is
        # domain-specific and implemented by the relevant subclasses
        raise NotImplementedError
