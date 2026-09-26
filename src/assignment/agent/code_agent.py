"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.agent.chess_tools import _invoke_skill
from assignment.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
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
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # Make the two coding tools available to the model, alongside any
        # invoke_skill tool that Agent registered for loaded skills.
        self.tools.append(EXECUTE_TOOL)
        self.tools.append(SEND_MESSAGE_TOOL)

        env = self.env
        environment_info = {
            "machine": getattr(env, "machine", "unknown"),
            "release": getattr(env, "release", "unknown"),
            "system": getattr(env, "system", "unknown"),
            "version": getattr(env, "version", "unknown"),
        }
        self.system_prompt = (
            "<system_information>\n"
            f"{json.dumps(environment_info, indent=2)}\n"
            "</system_information>\n"
            "\n"
            "You are a coding agent operating in the terminal described above. "
            "You are given a software task to solve: inspect the code, reproduce "
            "the problem, modify source files, and verify the fix by running "
            "tests or commands.\n"
            "\n"
            "Use the `execute` tool to run bash commands. After every result, "
            "reason in your message text about what it means, then take the next "
            "step. A non-zero exit code is a recoverable observation: read the "
            "output, adjust, and retry rather than giving up. Prefer commands "
            "that produce little output; when reading a file, use `head`, `tail`, "
            "or `sed -n` ranges instead of printing it all. Every command runs in "
            "a fresh subshell, so pass `cwd`/`env` arguments when a command needs "
            "a directory or environment.\n"
            "\n"
            "Work until the task is genuinely solved and verified, then send the "
            "user a concise summary of what you changed and the evidence it works "
            "with the `send_message` tool. Never claim success without having run "
            "the relevant tests or commands yourself."
        )
        if self.skills:
            catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += (
                "\n\nReusable skills are available. Call `invoke_skill` with a "
                "skill's name to load its instructions, and follow them in place "
                f"of your default approach.\n\n<skills>\n{catalog}\n</skills>\n"
            )
        working_directory = getattr(env, "cwd", "/")
        self.task_prompt = (
            f"Task: {self.task}\n\n"
            f"You are working in a terminal whose current directory is "
            f"{working_directory}. Solve the task there by running commands and "
            "editing files."
        )

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        observations: list[dict[str, str]] = []
        for call in tool_calls:
            call_id = call.get("id") or "unknown"
            function = call.get("function", {}) if isinstance(call, dict) else {}
            name = function.get("name")
            raw_arguments = function.get("arguments", "")

            # Malformed JSON becomes a recoverable observation: nothing runs and
            # the agent hears what went wrong instead of the run crashing.
            try:
                arguments = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else raw_arguments
                )
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": (
                            f"<tool_error>Malformed JSON arguments to "
                            f"`{name}`: {exc}. Nothing was executed.</tool_error>"
                        ),
                    }
                )
                continue

            if name == "execute":
                observations.append(
                    self._execute_command(call_id, arguments)
                )
            elif name == "send_message":
                summary = arguments.get("summary")
                if not isinstance(summary, str):
                    observations.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": (
                                "<tool_error>send_message requires a string "
                                "`summary` argument.</tool_error>"
                            ),
                        }
                    )
                else:
                    observations.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": f"Message sent to the user:\n{summary}",
                        }
                    )
                    # A message bound for the user is the agent's closing act.
                    self.finished = True
            elif name == "invoke_skill":
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _invoke_skill(self.skills, json.dumps(arguments)),
                    }
                )
            else:
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": (
                            f"<tool_error>Unknown tool `{name}`. "
                            "Nothing was executed.</tool_error>"
                        ),
                    }
                )
        return observations

    def _execute_command(
        self, call_id: str, arguments: dict[str, Any]
    ) -> dict[str, str]:
        """Run one ``execute`` call, forwarding any optional arguments."""

        command = arguments.get("command")
        if not isinstance(command, (str, list)):
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": (
                    "<tool_error>execute requires a string or list `command` "
                    "argument. Nothing was executed.</tool_error>"
                ),
            }
        try:
            result = self.env.execute(
                command=command,
                timeout=arguments.get("timeout"),
                cwd=arguments.get("cwd"),
                env=arguments.get("env"),
                shell=arguments.get("shell"),
            )
        except Exception as exc:  # noqa: BLE001 - relay, do not crash
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": (
                    f"<tool_error>The command could not be executed: "
                    f"{type(exc).__name__}: {exc}</tool_error>"
                ),
            }
        if not isinstance(result, dict):
            result = {"output": str(result), "returncode": None}
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": format_tool_output(result),
        }

