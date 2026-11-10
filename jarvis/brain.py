"""The brain: a persistent Claude Code process driven over stream-json.

It runs the unmodified `claude` CLI that the user is already signed in to with their
own Claude subscription, so no API key is used and no credentials are touched.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("jarvis.brain")

_END = object()  # sentinel: the process exited
_RETRY = object()  # the process died before producing anything for this turn

# Environment of the parent Claude Code session (if Jarvis was started from one) must not leak in.
_KEEP_CLAUDE_ENV = {"CLAUDE_CODE_GIT_BASH_PATH", "CLAUDE_CONFIG_DIR"}


@dataclass
class BrainSettings:
    model: str
    effort: Optional[str]
    system_prompt: str
    cwd: Path
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    exclude_claude_md: list[str] = field(default_factory=list)
    extra_settings: dict = field(default_factory=dict)  # e.g. hooks
    extra_env: dict = field(default_factory=dict)
    mcp_servers: dict = field(default_factory=dict)  # name -> stdio server config (e.g. the planner)


@dataclass
class TurnResult:
    text: str
    ok: bool
    error: Optional[str] = None
    interrupted: bool = False


def find_claude() -> str:
    exe = shutil.which("claude")
    if exe:
        return exe
    candidate = Path.home() / ".local" / "bin" / "claude.exe"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("Claude Code CLI (`claude`) не найден в PATH. Установите Claude Code и войдите в свой аккаунт.")


def build_command(exe: str, s: BrainSettings) -> list[str]:
    cmd = [
        exe, "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--model", s.model,
        "--append-system-prompt", s.system_prompt,
        # Only project settings: keep the user's global hooks/permissions out of Jarvis,
        # and no user MCP servers or skills either.
        "--setting-sources", "project",
        "--strict-mcp-config",  # only the servers below, never the user's own MCP servers
        "--disable-slash-commands",
        "--no-session-persistence",
    ]
    if s.effort:
        cmd += ["--effort", s.effort]
    if s.mcp_servers:
        cmd += ["--mcp-config", json.dumps({"mcpServers": s.mcp_servers}, ensure_ascii=False)]
    settings: dict = dict(s.extra_settings)
    if s.exclude_claude_md:
        settings["claudeMdExcludes"] = s.exclude_claude_md
    if settings:
        cmd += ["--settings", json.dumps(settings, ensure_ascii=False)]
    if s.allowed_tools:
        cmd += ["--allowedTools", ",".join(s.allowed_tools)]
    if s.disallowed_tools:
        cmd += ["--disallowedTools", ",".join(s.disallowed_tools)]
    return cmd


def child_env(base: dict, extra: dict) -> dict:
    env = {
        k: v for k, v in base.items()
        if not ((k == "CLAUDECODE" or k.startswith("CLAUDE_CODE_")) and k not in _KEEP_CLAUDE_ENV)
    }
    # Never let a stray API key silently switch billing away from the subscription.
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    env.update(extra)
    return env


def tool_summary(name: str, tool_input: dict) -> str:
    """A short human-readable description of a tool call for the UI log."""
    for key in ("command", "file_path", "query", "url", "pattern", "path"):
        if key in tool_input and tool_input[key]:
            value = str(tool_input[key]).replace("\n", " ")
            return f"{name}: {value[:160]}"
    return name


class ClaudeBrain:
    def __init__(self, settings: BrainSettings, stop_grace: float = 4.0):
        self.settings = settings
        self.stop_grace = stop_grace
        self._exe = find_claude()
        self._proc: Optional[subprocess.Popen] = None
        self._events: "queue.Queue[object]" = queue.Queue()
        self._turn_lock = threading.Lock()
        self._proc_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._interrupted = threading.Event()
        self._interrupt_at = 0.0
        self._in_turn = False
        self.model_name: Optional[str] = None
        self.auth_source: Optional[str] = None
        self.fresh = True  # True until the current process has answered once

    # ---- process lifecycle -------------------------------------------------
    def _start_locked(self) -> None:
        self.settings.cwd.mkdir(parents=True, exist_ok=True)
        cmd = build_command(self._exe, self.settings)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        log.info("starting claude: model=%s cwd=%s", self.settings.model, self.settings.cwd)
        events: "queue.Queue[object]" = queue.Queue()
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.settings.cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_env(dict(os.environ), self.settings.extra_env),
            creationflags=creationflags,
        )
        self._proc, self._events, self.fresh = proc, events, True
        threading.Thread(target=self._read_stdout, args=(proc, events), daemon=True, name="claude-stdout").start()
        threading.Thread(target=self._read_stderr, args=(proc,), daemon=True, name="claude-stderr").start()

    def _ensure_running(self) -> None:
        with self._proc_lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start_locked()

    @staticmethod
    def _read_stdout(proc: subprocess.Popen, events: "queue.Queue[object]") -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    log.debug("non-json stdout: %s", line[:200])
                    continue
                if isinstance(obj, dict):
                    events.put(obj)
        finally:
            events.put(_END)

    @staticmethod
    def _read_stderr(proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            if line.strip():
                log.warning("claude stderr: %s", line.rstrip()[:300])

    def _send(self, obj: dict) -> None:
        with self._write_lock:
            proc = self._proc
            if proc is None or proc.stdin is None:
                raise OSError("claude is not running")
            proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            proc.stdin.flush()

    def warm_up(self) -> None:
        """Start the process ahead of the first question to hide start-up latency."""
        self._ensure_running()

    def close(self, kill: bool = False) -> None:
        with self._proc_lock:
            proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            try:
                if kill:
                    proc.kill()
                else:
                    if proc.stdin:
                        proc.stdin.close()
                    proc.wait(timeout=5)
            except Exception:
                proc.kill()

    # ---- conversation ------------------------------------------------------
    def interrupt(self) -> None:
        """Ask Claude to stop the current turn. If it does not, the turn is killed after stop_grace."""
        if not self._in_turn:
            return
        self._interrupt_at = time.monotonic()
        self._interrupted.set()
        try:
            self._send({
                "type": "control_request",
                "request_id": f"int_{uuid.uuid4().hex[:12]}",
                "request": {"subtype": "interrupt"},
            })
        except Exception as exc:  # the deadline in _collect still applies
            log.debug("interrupt send failed: %s", exc)

    def ask(
        self,
        text: str,
        on_delta: Callable[[str], None] = lambda _t: None,
        on_tool: Callable[[str], None] = lambda _t: None,
        fresh_prefix: Optional[Callable[[], str]] = None,
        cancel: Optional[threading.Event] = None,
        timeout: float = 300.0,
    ) -> TurnResult:
        with self._turn_lock:
            self._interrupted.clear()
            for attempt in (1, 2):
                if cancel is not None and cancel.is_set():
                    return TurnResult(text="", ok=True, interrupted=True)
                try:
                    self._ensure_running()
                except OSError as exc:
                    return TurnResult(text="", ok=False, error=f"Не удалось запустить Claude: {exc}")
                prefix = fresh_prefix() if (self.fresh and fresh_prefix) else ""
                events = self._events
                self._in_turn = True
                try:
                    try:
                        self._send({"type": "user", "message": {"role": "user", "content": prefix + text}})
                    except (OSError, ValueError) as exc:
                        log.warning("send failed (attempt %d): %s", attempt, exc)
                        self.close(kill=True)
                        continue
                    result = self._collect(events, on_delta, on_tool, timeout)
                finally:
                    self._in_turn = False
                if result is _RETRY:
                    log.warning("claude exited before answering (attempt %d)", attempt)
                    self.close(kill=True)
                    continue
                assert isinstance(result, TurnResult)
                if result.ok and not result.interrupted:
                    self.fresh = False
                return result
            return TurnResult(text="", ok=False, error="Процесс Claude завершился. Проверьте, что вы вошли в Claude Code (`claude` → /login).")

    def _collect(self, events, on_delta, on_tool, timeout) -> object:
        parts: list[str] = []
        streamed_any = False
        produced = False  # text or tool use seen: retrying would repeat side effects
        last_event = time.monotonic()
        while True:
            now = time.monotonic()
            if self._interrupted.is_set() and now - self._interrupt_at > self.stop_grace:
                # Claude ignored the interrupt (e.g. a hung command): kill it so nothing desyncs.
                log.warning("claude did not stop in %.0fs, restarting it", self.stop_grace)
                self.close(kill=True)
                return TurnResult(text="".join(parts).strip(), ok=True, interrupted=True)
            if now - last_event > timeout:
                self.close(kill=True)
                return TurnResult(text="".join(parts).strip(), ok=False, error="Claude не ответил вовремя.")
            try:
                ev = events.get(timeout=0.25)
            except queue.Empty:
                continue
            last_event = time.monotonic()
            if ev is _END:
                if self._interrupted.is_set():
                    return TurnResult(text="".join(parts).strip(), ok=True, interrupted=True)
                if not produced:
                    return _RETRY
                return TurnResult(text="".join(parts).strip(), ok=False, error="Процесс Claude неожиданно завершился.")
            etype = ev.get("type")
            if etype == "system" and ev.get("subtype") == "init":
                self.model_name = ev.get("model")
                self.auth_source = ev.get("apiKeySource")
            elif etype == "stream_event":
                inner = ev.get("event") or {}
                if inner.get("type") == "content_block_delta":
                    delta = inner.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        streamed_any = produced = True
                        parts.append(delta["text"])
                        if not self._interrupted.is_set():
                            on_delta(delta["text"])
                elif inner.get("type") == "message_start" and parts and not self._interrupted.is_set():
                    # A new assistant message after tool use: keep sentences apart.
                    parts.append(" ")
                    on_delta(" ")
            elif etype == "assistant":
                for block in (ev.get("message") or {}).get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        produced = True
                        on_tool(tool_summary(block.get("name", "tool"), block.get("input") or {}))
                    elif block.get("type") == "text" and not streamed_any:
                        # Fallback if partial messages are unavailable.
                        produced = True
                        parts.append(block.get("text", ""))
                        if not self._interrupted.is_set():
                            on_delta(block.get("text", ""))
            elif etype == "result":
                text = "".join(parts).strip()
                if self._interrupted.is_set():
                    return TurnResult(text=text, ok=True, interrupted=True)
                if ev.get("is_error"):
                    err = ev.get("result") or ev.get("subtype") or "неизвестная ошибка"
                    return TurnResult(text=text, ok=False, error=str(err))
                if not text:
                    text = str(ev.get("result") or "").strip()
                return TurnResult(text=text, ok=True)
