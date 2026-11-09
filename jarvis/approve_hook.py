"""Claude Code PreToolUse hook: asks the running Jarvis whether a shell command or web fetch may run.

Claude Code passes the tool call as JSON on stdin; we forward it to Jarvis on
127.0.0.1 (authenticated with a per-run token) and print the decision.
Any failure means "deny": a command never runs without an answer.
Standard library only, so it starts fast.
"""
import json
import os
import sys
import urllib.request


class _Decided(Exception):
    pass


def decide(decision: str, reason: str) -> None:
    out = json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }, ensure_ascii=True)  # ASCII: printable under any console code page
    sys.stdout.write(out + "\n")
    sys.stdout.flush()
    raise _Decided


def main() -> None:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise ValueError("payload")
    except Exception:
        decide("deny", "Джарвис не смог прочитать запрос.")
    url = os.environ.get("JARVIS_APPROVE_URL")
    token = os.environ.get("JARVIS_TOKEN")
    if not url or not token:
        decide("deny", "Нет связи с Джарвисом для подтверждения.")
    body = json.dumps({
        "tool_name": payload.get("tool_name", ""),
        "tool_input": payload.get("tool_input") or {},
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json", "X-Jarvis-Token": token,
    })
    try:
        with urllib.request.urlopen(req, timeout=125) as resp:
            answer = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        decide("deny", f"Подтверждение не получено ({type(exc).__name__}).")
    if not isinstance(answer, dict):
        decide("deny", "Bad answer from Jarvis.")
    if answer.get("decision") == "allow":
        decide("allow", answer.get("reason") or "Разрешено пользователем.")
    decide("deny", answer.get("reason") or "Пользователь не разрешил эту команду.")


if __name__ == "__main__":
    try:
        main()
    except _Decided:
        sys.exit(0)
    except BaseException:  # anything unexpected: block the command (exit 2 = deny in Claude Code)
        sys.stderr.write("Jarvis approval hook failed: command blocked.\n")
        sys.exit(2)
    sys.exit(2)  # main() always decides; reaching here means it did not
