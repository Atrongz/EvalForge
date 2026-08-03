"""Target adapters — the systems under test.

One uniform contract, one file per kind, one line in the registry. Adding a new
kind of system to evaluate should never require touching the runner, the
scorers, or the reporter.

A Target takes a case input and returns a TargetResponse. That's the whole
interface. Whether it shells out to a CLI, POSTs to an endpoint, imports a
Python callable, or drives a subscription-backed model is the adapter's problem.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class TargetResponse:
    """What a target gave back."""

    output: Any
    input_tokens: int = 0
    output_tokens: int = 0
    # Free-form; surfaced in the report so a run is traceable to what produced it.
    meta: dict[str, Any] = field(default_factory=dict)


class Target(Protocol):
    id: str

    def version(self) -> str:
        """Identifies what's under test — model id, git sha, endpoint. Recorded
        on every run so a regression can be attributed to a change."""
        ...

    def invoke(self, case_input: Any) -> TargetResponse: ...


# ---------------------------------------------------------------------------


class PythonTarget:
    """Call a Python callable: `target_config: {ref: "mypkg.mod:func"}`.

    The zero-dependency path. Use for anything importable — deterministic
    business logic, a local pipeline, a wrapper you write yourself.
    """

    id = "python"

    def __init__(self, ref: str, **_: Any) -> None:
        if ":" not in ref:
            raise ValueError(f"python target ref must be 'module:attr', got {ref!r}")
        module_name, attr = ref.split(":", 1)
        self._ref = ref
        module = importlib.import_module(module_name)
        fn = getattr(module, attr, None)
        if not callable(fn):
            raise ValueError(f"{ref!r} is not callable")
        self._fn: Callable[[Any], Any] = fn

    def version(self) -> str:
        return self._ref

    def invoke(self, case_input: Any) -> TargetResponse:
        return TargetResponse(output=self._fn(case_input))


class CommandTarget:
    """Shell out: `target_config: {command: ["python", "run.py"], json: true}`.

    The case input is written to stdin. stdout is the output — parsed as JSON
    when `json: true`, otherwise returned as text. Non-zero exit raises, so a
    crashed target reads as an error rather than silently scoring zero.
    """

    id = "command"

    def __init__(
        self,
        command: list[str],
        json: bool = False,
        timeout: int = 120,
        cwd: str | None = None,
        **_: Any,
    ) -> None:
        if not command:
            raise ValueError("command target requires a non-empty 'command'")
        self._command = command
        self._json = json
        self._timeout = timeout
        self._cwd = cwd

    def version(self) -> str:
        return " ".join(self._command)

    def invoke(self, case_input: Any) -> TargetResponse:
        payload = (
            case_input if isinstance(case_input, str) else json_dumps(case_input)
        )
        proc = subprocess.run(
            self._command,
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self._timeout,
            cwd=self._cwd,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"target exited {proc.returncode}: {proc.stderr.strip()[:400]}"
            )
        out = proc.stdout.strip()
        return TargetResponse(output=_maybe_json(out) if self._json else out)


class HttpTarget:
    """POST the case input as JSON to an endpoint.

    `target_config: {url: "http://localhost:8000/rag/ask", field: "answer"}`
    `field` optionally pulls one key out of the JSON response.
    """

    id = "http"

    def __init__(
        self,
        url: str,
        field: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: int = 120,
        **_: Any,
    ) -> None:
        self._url = url
        self._field = field
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._timeout = timeout

    def version(self) -> str:
        return self._url

    def invoke(self, case_input: Any) -> TargetResponse:
        import urllib.request

        body = json_dumps(
            case_input if isinstance(case_input, dict) else {"input": case_input}
        ).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=body, headers=self._headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            raw = resp.read().decode("utf-8")
        data = _maybe_json(raw)
        if self._field and isinstance(data, dict):
            data = data.get(self._field)
        return TargetResponse(output=data)


def claude_executable() -> str:
    """Absolute path to the Claude CLI.

    Must be the resolved path, not the bare name: on Windows the CLI is a
    `claude.CMD` shim and CreateProcess cannot launch that from a bare name —
    it fails with WinError 2. Resolving here also keeps `shell=True` off the
    table, which matters because this harness feeds deliberately adversarial
    text through these arguments.
    """
    path = shutil.which("claude")
    if not path:
        raise RuntimeError(
            "claude CLI not found on PATH — install it or use the 'anthropic_api' target"
        )
    return path


class ClaudeCliTarget:
    """Drive the Claude CLI under the logged-in subscription — no API key, $0.

    `target_config: {system: "...", model: "sonnet"}`

    Deliberately the default for prompt-level suites: an eval harness you can't
    afford to run is an eval harness you won't run. Token counts aren't reported
    by the CLI, so the cost column reads zero here — which is accurate.
    """

    id = "claude_cli"

    def __init__(
        self,
        system: str = "",
        model: str = "",
        timeout: int = 180,
        **_: Any,
    ) -> None:
        self._exe = claude_executable()
        self._system = system
        self._model = model
        self._timeout = timeout

    def version(self) -> str:
        return f"claude-cli:{self._model or 'default'}"

    def invoke(self, case_input: Any) -> TargetResponse:
        prompt = case_input if isinstance(case_input, str) else json_dumps(case_input)
        # The prompt goes on STDIN, never in argv. On Windows the CLI is a .CMD
        # shim, so argv is routed through cmd.exe and a multi-line prompt is
        # silently truncated at the first newline — the target answers a
        # question you never asked and the run looks valid. Verified 2026-08-03:
        # via argv only line 1 arrives; via stdin the whole prompt arrives.
        cmd = [self._exe, "-p"]
        if self._model:
            cmd += ["--model", self._model]
        if self._system:
            cmd += ["--append-system-prompt", self._system]
        env = {**os.environ}
        # Force the subscription path — a stray key would silently bill the API.
        env.pop("ANTHROPIC_API_KEY", None)
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self._timeout,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude cli exited {proc.returncode}: {proc.stderr[:400]}")
        return TargetResponse(output=proc.stdout.strip())


class AnthropicApiTarget:
    """Metered API path. Reports real token usage so the cost column is honest."""

    id = "anthropic_api"

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        system: str = "",
        max_tokens: int = 2048,
        **_: Any,
    ) -> None:
        import anthropic  # imported lazily — optional dependency

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic()
        self._model = model
        self._system = system
        self._max_tokens = max_tokens

    def version(self) -> str:
        return f"anthropic-api:{self._model}"

    def invoke(self, case_input: Any) -> TargetResponse:
        prompt = case_input if isinstance(case_input, str) else json_dumps(case_input)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._system:
            kwargs["system"] = self._system
        msg = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return TargetResponse(
            output=text,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )


class EchoTarget:
    """Returns the case's own `expected`. Used by --dry-run and by the test suite
    to exercise runner/scorer/report wiring with no external calls."""

    id = "echo"

    def __init__(self, **_: Any) -> None:
        pass

    def version(self) -> str:
        return "echo"

    def invoke(self, case_input: Any) -> TargetResponse:
        return TargetResponse(output=case_input)


# ---------------------------------------------------------------------------
# Registry — one line per adapter. Nothing else changes when you add one.
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {
    "python": PythonTarget,
    "command": CommandTarget,
    "http": HttpTarget,
    "claude_cli": ClaudeCliTarget,
    "anthropic_api": AnthropicApiTarget,
    "echo": EchoTarget,
}


def build_target(kind: str, config: dict[str, Any]) -> Target:
    if kind not in _REGISTRY:
        raise ValueError(
            f"unknown target {kind!r}. Registered: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[kind](**config)  # type: ignore[return-value]


def registered_targets() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _maybe_json(text: str) -> Any:
    """Parse JSON when it is JSON; otherwise hand back the text unchanged.

    Tolerates a model wrapping its JSON in a ```json fence, which is the single
    most common shape of 'valid answer, unparseable output'.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return text
