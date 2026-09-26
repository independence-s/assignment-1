"""Chess tool implementations, decoupled from the agent that registers them.

Every function here takes the HTTP client explicitly instead of reading it off
an agent, so the same code can run in the agent process or inside the sandbox
beside the server it talks to.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

CHESS_PORT = 8000


def _request_state(
    client: httpx.Client, method: str, endpoint: str, **kwargs: Any
) -> dict[str, Any]:
    """Make one chess API request and validate its JSON response."""

    response = client.request(method, endpoint, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Chess server returned non-JSON ({response.status_code})."
        ) from exc
    if response.status_code >= 400:
        detail = (
            payload.get("detail", payload) if isinstance(payload, dict) else payload
        )
        raise ValueError(str(detail))
    if not isinstance(payload, dict):
        raise RuntimeError("Chess server response must be a JSON object.")
    return payload


def _simulate_move(client: httpx.Client, arguments: str) -> str:
    """New tool: inspect FEN or simulate one ply without changing the game.

    Takes the raw JSON arguments of one tool call and returns the observation
    to send back, so a bad argument or a server error reaches the model as a
    recoverable ``<chess_error>`` instead of ending the run.
    """
    try:
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("arguments must be a JSON object")
        fen = parsed.get("fen")
        if not isinstance(fen, str) or not fen:
            raise ValueError("simulate_move requires a string `fen` argument")
        move = parsed.get("move")
        if move is not None and not isinstance(move, str):
            raise ValueError("`move` must be a string when provided")
        payload = {"fen": fen}
        if move is not None:
            payload["move"] = move
        state = _request_state(client, "POST", "/api/simulate", json=payload)
        return json.dumps(state)
    except Exception as exc:  # noqa: BLE001 - recoverable observation
        return f"<chess_error>simulate_move failed: {exc}</chess_error>"


def _play_move(client: httpx.Client, arguments: str) -> str:
    """Existing tool: play one move as White and return the resulting state.

    Takes the raw JSON arguments of one tool call. Returns the new state, or a
    `<chess_error>` observation if the move could not be played.
    """
    try:
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("arguments must be a JSON object")
        move = parsed.get("move")
        if not isinstance(move, str) or not move:
            raise ValueError("play_move requires a string `move` argument")
        state = _request_state(
            client, "POST", "/api/move", json={"move": move}
        )
        return json.dumps(state)
    except Exception as exc:  # noqa: BLE001 - recoverable observation
        return f"<chess_error>play_move failed: {exc}</chess_error>"


def _run_python(env: Any, port: int, arguments: str) -> str:
    """New tool: run Python with access to the existing registered tools.

    The snippet runs inside the sandbox, which already has the tool
    implementations and the chess server, so code the model wrote never
    executes in the agent process.
    """
    try:
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("arguments must be a JSON object")
        code = parsed.get("code")
        if not isinstance(code, str) or not code:
            raise ValueError("run_python requires a string `code` argument")
    except Exception as exc:  # noqa: BLE001 - recoverable observation
        return f"<chess_error>run_python arguments invalid: {exc}</chess_error>"

    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    command = f"python /opt/assignment/sandbox_python.py {port} {encoded}"
    try:
        result = env.execute(command)
    except Exception as exc:  # noqa: BLE001 - sandbox transport failure
        return f"<chess_error>run_python could not run in the sandbox: {exc}</chess_error>"
    if not isinstance(result, dict):
        return f"<chess_error>run_python returned an unexpected result.</chess_error>"

    output = result.get("output")
    if output is None:
        stderr = result.get("stderr") or ""
        output = (result.get("stdout") or "") + (stderr if stderr else "")
    if result.get("returncode") not in (0, None):
        message = result.get("exception_info") or output or "non-zero exit code"
        return f"<chess_error>sandbox runner failed: {message}</chess_error>"
    # The runner prints one JSON object on stdout; relay it verbatim.
    return str(output)


def _invoke_skill(skills: dict[str, dict[str, str]], arguments: str) -> str:
    """Existing tool: load one skill's instructions into the conversation."""
    try:
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("arguments must be a JSON object")
        name = parsed.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("invoke_skill requires a string `name` argument")
    except Exception as exc:  # noqa: BLE001 - recoverable observation
        return f"<chess_error>invoke_skill arguments invalid: {exc}</chess_error>"
    if name not in skills:
        available = ", ".join(sorted(skills)) or "(none loaded)"
        return (
            f"<chess_error>No skill named '{name}'. "
            f"Available skills: {available}</chess_error>"
        )
    return skills[name]["content"]


def _game_state(client: httpx.Client, reset: bool = False) -> dict:
    """Read the live game, or start a new one and read the opening position."""

    method, endpoint = ("POST", "/api/reset") if reset else ("GET", "/api/state")
    return _request_state(client, method, endpoint)
