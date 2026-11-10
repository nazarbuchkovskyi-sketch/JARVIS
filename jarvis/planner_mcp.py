"""MCP server (stdio) that gives Jarvis's Claude read-only tools over the Planning System.

Started by Claude Code itself (see core.Jarvis._brain_settings) as
    python -I <repo>/jarvis/planner_mcp.py
It reads the connection from %APPDATA%/Jarvis/planner.env, so no secret is passed on a command line or
through Claude. Plain JSON-RPC 2.0 over newline-delimited stdio; no MCP SDK needed.
Only reads: there is no tool that changes anything.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional

if __package__ in (None, ""):  # run as a script with -I: make `jarvis` importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.planner import (  # noqa: E402
    MAX_RANGE_DAYS, PlannerDB, PlannerError, header_line, load_settings, now_and_next, parse_day, planner_day,
)

log = logging.getLogger("jarvis.planner_mcp")

SERVER_NAME = "planner"
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
_DATE = {"type": "string", "description": "YYYY-MM-DD, or today / tomorrow / yesterday"}
_RO = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}

TOOLS: list[dict] = [
    {
        "name": "planner_day",
        "description": "The user's plan for one day from the Planning System: calendar time blocks "
                       "(time, title, done) and open to-dos for that day. Default: today. Read-only.",
        "inputSchema": {"type": "object", "properties": {"date": _DATE}, "additionalProperties": False},
    },
    {
        "name": "planner_upcoming",
        "description": "What is next: the user's calendar blocks from now on, plus what is running right now. Read-only.",
        "inputSchema": {"type": "object", "properties": {
            "hours": {"type": "integer", "minimum": 1, "maximum": 72, "description": "look ahead this many hours (default 12)"},
        }, "additionalProperties": False},
    },
    {
        "name": "planner_range",
        "description": f"The user's calendar blocks for a date range (at most {MAX_RANGE_DAYS} days), e.g. the week. Read-only.",
        "inputSchema": {"type": "object", "properties": {"from": _DATE, "to": _DATE},
                        "required": ["from", "to"], "additionalProperties": False},
    },
    {
        "name": "planner_search",
        "description": "Find calendar blocks whose title contains a word (e.g. 'созвон', 'Влад'), "
                       "by default from 7 days ago to 60 days ahead. Read-only.",
        "inputSchema": {"type": "object", "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 100},
            "days_back": {"type": "integer", "minimum": 0, "maximum": 365},
            "days_ahead": {"type": "integer", "minimum": 0, "maximum": 365},
        }, "required": ["query"], "additionalProperties": False},
    },
    {
        "name": "planner_todos",
        "description": "All open to-dos of the user (unscheduled items under the planner: work, personal, reminder), "
                       "with their dates and which are overdue. Read-only.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "planner_tasks",
        "description": "Open team tasks (kanban) assigned to the user: status, priority, due date, project. Read-only.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "planner_week_focus",
        "description": "The week plan: focus of the week and its top outcomes (done or not). Default: this week. Read-only.",
        "inputSchema": {"type": "object", "properties": {"date": _DATE}, "additionalProperties": False},
    },
]
for _t in TOOLS:
    _t["annotations"] = {"title": _t["name"], **_RO}


class Handlers:
    def __init__(self, db: Optional[PlannerDB], setup_error: Optional[str] = None):
        self.db = db
        self.setup_error = setup_error

    def _db(self) -> PlannerDB:
        if self.db is None:
            raise PlannerError(self.setup_error or "Планер не подключён: нет PLANNER_DATABASE_URL (python -m jarvis.planner_setup).")
        self.db.user()  # resolves the user and their time zone
        return self.db

    def call(self, name: str, args: dict) -> dict:
        fn: Optional[Callable[[dict], dict]] = getattr(self, f"t_{name}", None)
        if fn is None or name not in {t["name"] for t in TOOLS}:
            raise PlannerError(f"Неизвестный инструмент {name}")
        return fn(args or {})

    def _base(self, db: PlannerDB) -> dict:
        now = db.now()
        return {"now": now.strftime("%Y-%m-%d %H:%M"), "planner_today": planner_day(now).isoformat(),
                "user": db.user()["name"]}

    def t_planner_day(self, a: dict) -> dict:
        db = self._db()
        now = db.now()
        day = parse_day(a.get("date"), now)
        events = db.events(day, day)
        out = {**self._base(db), "date": day.isoformat(), "weekday": _weekday(day),
               "calendar": [e.as_dict() for e in events], "todos": db.todos(day)}
        if day == planner_day(now):
            out["now_line"] = header_line(events, now)
        return out

    def t_planner_upcoming(self, a: dict) -> dict:
        db = self._db()
        now = db.now()
        hours = max(1, min(72, int(a.get("hours") or 12)))
        today = planner_day(now)
        until = now + dt.timedelta(hours=hours)
        events = db.events(today, (until - dt.timedelta(minutes=60)).date())
        current, _ = now_and_next(events, now)
        nxt = [e.as_dict() for e in events if not e.done and now < e.start <= until]
        return {**self._base(db), "hours": hours, "now_running": current.as_dict() if current else None, "next": nxt}

    def t_planner_range(self, a: dict) -> dict:
        db = self._db()
        now = db.now()
        start, end = parse_day(a.get("from"), now), parse_day(a.get("to"), now)
        events = db.events(start, end)
        days: dict[str, list] = {}
        for e in events:
            days.setdefault(e.date.isoformat(), []).append(e.as_dict())
        return {**self._base(db), "from": min(start, end).isoformat(), "to": max(start, end).isoformat(),
                "days": [{"date": d, "weekday": _weekday(dt.date.fromisoformat(d)), "calendar": v}
                         for d, v in sorted(days.items())]}

    def t_planner_search(self, a: dict) -> dict:
        db = self._db()
        today = planner_day(db.now())
        back = max(0, min(365, int(a.get("days_back", 7))))
        ahead = max(0, min(365, int(a.get("days_ahead", 60))))
        start, end = today - dt.timedelta(days=back), today + dt.timedelta(days=ahead)
        found = db.search(str(a.get("query", "")), start, end)
        return {**self._base(db), "query": a.get("query"), "from": start.isoformat(), "to": end.isoformat(),
                "found": [e.as_dict() for e in found]}

    def t_planner_todos(self, a: dict) -> dict:
        db = self._db()
        today = planner_day(db.now())
        items = db.todos()
        for i in items:
            if i["date"] and i["date"] < today.isoformat():
                i["overdue"] = True
        return {**self._base(db), "todos": items}

    def t_planner_tasks(self, a: dict) -> dict:
        db = self._db()
        today = planner_day(db.now()).isoformat()
        tasks = db.tasks()
        for t in tasks:
            if t["due"] and t["due"] < today:
                t["overdue"] = True
        return {**self._base(db), "tasks": tasks}

    def t_planner_week_focus(self, a: dict) -> dict:
        db = self._db()
        day = parse_day(a.get("date"), db.now())
        return {**self._base(db), "week": db.week_focus(day)}


_WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]


def _weekday(d: dt.date) -> str:
    return _WEEKDAYS[d.weekday()]


# ---- JSON-RPC over stdio ---------------------------------------------------
def handle_message(msg: Any, handlers: Handlers) -> Optional[dict]:
    """Return the response for one JSON-RPC message (None for notifications)."""
    if not isinstance(msg, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:  # a notification (initialized, cancelled, ...): nothing to answer
        return None
    params = msg.get("params") or {}

    def ok(result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    if method == "initialize":
        asked = params.get("protocolVersion") if isinstance(params, dict) else None
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return ok({
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "jarvis-planner", "version": "1.0.0"},
            "instructions": "Read-only access to the user's Planning System (calendar, to-dos, tasks, week focus). "
                            "Nothing can be changed through these tools.",
        })
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name") if isinstance(params, dict) else None
        args = params.get("arguments") if isinstance(params, dict) else None
        if name not in {t["name"] for t in TOOLS}:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"Unknown tool: {name}"}}
        try:
            data = handlers.call(str(name), args if isinstance(args, dict) else {})
            return ok({"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}], "isError": False})
        except PlannerError as exc:
            return ok({"content": [{"type": "text", "text": str(exc)}], "isError": True})
        except (TypeError, ValueError) as exc:
            return ok({"content": [{"type": "text", "text": f"Неверные параметры: {exc}"}], "isError": True})
        except Exception as exc:  # keep the server alive; never leak a traceback with secrets
            log.exception("tool %s failed", name)
            return ok({"content": [{"type": "text", "text": f"Ошибка планера ({type(exc).__name__})."}], "isError": True})
    if method in ("resources/list", "prompts/list"):
        return ok({"resources": []} if method == "resources/list" else {"prompts": []})
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def serve(stdin=None, stdout=None, handlers: Optional[Handlers] = None) -> None:
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout.buffer
    if handlers is None:
        try:
            settings = load_settings()
            handlers = Handlers(PlannerDB(settings) if settings else None)
        except PlannerError as exc:
            handlers = Handlers(None, str(exc))
    for raw in stdin:
        line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            resp: Optional[dict] = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        else:
            if isinstance(msg, list):  # batch (old protocol versions)
                out = [r for r in (handle_message(m, handlers) for m in msg) if r is not None]
                resp = out or None  # type: ignore[assignment]
            else:
                resp = handle_message(msg, handlers)
        if resp is not None:
            # ASCII JSON: safe under any console code page.
            stdout.write((json.dumps(resp, ensure_ascii=True) + "\n").encode("ascii"))
            stdout.flush()
    if handlers.db is not None:
        handlers.db.close()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="planner-mcp %(levelname)s: %(message)s")
    serve()


if __name__ == "__main__":
    main()
