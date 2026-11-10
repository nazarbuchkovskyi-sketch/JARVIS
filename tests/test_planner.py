"""Planner integration: pure logic, the MCP protocol and the read-only guarantees (no network)."""
import datetime as dt
import io
import json
import random

import pytest

from jarvis import persona
from jarvis.brain import BrainSettings, build_command
from jarvis.planner import (
    Event, PlannerError, PlannerSettings, _like_pattern, _reason, due_reminders, header_line, hhmm,
    load_settings, minutes_phrase, now_and_next, parse_day, planner_day, read_env_file, reminder_text, still_due,
)
from jarvis.planner_mcp import TOOLS, Handlers, handle_message, serve
from jarvis.planner_setup import readonly_url, setup_sql, write_env

D = dt.date(2026, 11, 10)


def ev(title, start, end, done=False, day=D, eid=None):
    return Event(eid or title, title, day, start, end, done)


# ---- settings --------------------------------------------------------------
def test_env_file_parsing(tmp_path):
    p = tmp_path / ".env"
    p.write_text('﻿# comment\nexport PLANNER_DATABASE_URL="postgresql://u:p@h:1/db"\n\nPLANNER_USER = Vlad \nX\n',
                 encoding="utf-8")
    assert read_env_file(p) == {"PLANNER_DATABASE_URL": "postgresql://u:p@h:1/db", "PLANNER_USER": "Vlad"}
    s = load_settings(p)
    assert s.user == "Vlad" and s.host == "h"
    assert "p@" not in repr(s)  # the password never shows up in logs


def test_settings_absent_or_wrong(tmp_path):
    assert load_settings(tmp_path / "missing.env") is None
    p = tmp_path / ".env"
    p.write_text("PLANNER_DATABASE_URL=mysql://x\n", encoding="utf-8")
    with pytest.raises(PlannerError):
        load_settings(p)


def test_reason_hides_secrets():
    r = _reason(Exception('connection to postgresql://jarvis_ro:s3cret@h/db failed: password authentication'))
    assert "s3cret" not in r


# ---- calendar maths --------------------------------------------------------
def test_planner_day_starts_at_one_am():
    assert planner_day(dt.datetime(2026, 11, 10, 0, 30)) == dt.date(2026, 11, 9)
    assert planner_day(dt.datetime(2026, 11, 10, 1, 0)) == D


def test_event_after_midnight_belongs_to_previous_column():
    e = ev("ночь", 1470, 1500)  # 00:30–01:00 after the 10th
    assert e.start == dt.datetime(2026, 11, 11, 0, 30)
    assert e.as_dict()["time"] == "00:30–01:00" and e.as_dict()["after_midnight"]
    assert hhmm(1440) == "00:00"


def test_parse_day():
    now = dt.datetime(2026, 11, 10, 14, 18)
    assert parse_day(None, now) == D
    assert parse_day("tomorrow", now) == dt.date(2026, 11, 11)
    assert parse_day("вчера", now) == dt.date(2026, 11, 9)
    assert parse_day("2026-12-01", now) == dt.date(2026, 12, 1)
    with pytest.raises(PlannerError):
        parse_day("next friday", now)


def test_now_next_and_header():
    now = dt.datetime(2026, 11, 10, 14, 18)
    events = [ev("Rest", 720, 750, done=True), ev("Jarvis", 840, 887), ev("Все для мініка", 887, 932),
              ev("Завтра", 600, 660, day=D + dt.timedelta(days=1))]
    cur, nxt = now_and_next(events, now)
    assert cur.title == "Jarvis" and [e.title for e in nxt] == ["Все для мініка"]
    line = header_line(events, now)
    assert "сейчас «Jarvis» до 14:47" in line and "14:47 «Все для мініка»" in line
    assert header_line(events, dt.datetime(2026, 11, 10, 23, 0)).startswith("план на сегодня: больше ничего")


# ---- reminders -------------------------------------------------------------
def test_reminder_window_and_dedup():
    e = ev("Созвон с Владом", 900, 960)  # 15:00
    sent: set = set()
    assert due_reminders([e], dt.datetime(2026, 11, 10, 14, 49), [10], sent) == []
    due = due_reminders([e], dt.datetime(2026, 11, 10, 14, 50), [10], sent)
    assert len(due) == 1 and due[0].lead == 10
    sent.update(r.key for r in due)
    assert due_reminders([e], dt.datetime(2026, 11, 10, 14, 55), [10], sent) == []  # spoken once
    # Jarvis started late (14:57): still reminded, with the real time left.
    late = due_reminders([e], dt.datetime(2026, 11, 10, 14, 57), [10], set())
    assert reminder_text(late, dt.datetime(2026, 11, 10, 14, 57), "шеф") == \
        "Шеф, через три минуты по плану: «Созвон с Владом»."
    assert due_reminders([e], dt.datetime(2026, 11, 10, 15, 0), [10], set()) == []  # already started


def test_reminder_at_start_and_done_events():
    e = ev("Jarvis", 840, 887)
    at = due_reminders([e], dt.datetime(2026, 11, 10, 14, 1), [10, 0], set())
    assert [r.lead for r in at] == [0]
    assert reminder_text(at, dt.datetime(2026, 11, 10, 14, 1), "сэр") == "Сэр, по плану сейчас: «Jarvis»."
    assert not still_due(at[0], dt.datetime(2026, 11, 10, 14, 4))
    assert due_reminders([ev("done", 840, 900, done=True)], dt.datetime(2026, 11, 10, 13, 55), [10], set()) == []
    # Rescheduled block (new start) is reminded again.
    sent = {r.key for r in due_reminders([e], dt.datetime(2026, 11, 10, 13, 55), [10], set())}
    moved = ev("Jarvis", 870, 900)
    assert due_reminders([moved], dt.datetime(2026, 11, 10, 14, 25), [10], sent)


def test_reminders_together_in_one_sentence():
    now = dt.datetime(2026, 11, 10, 14, 42)
    rs = due_reminders([ev("A", 887, 900, eid="1"), ev("B", 887, 900, eid="2")], now, [10], set())
    assert reminder_text(rs, now, "") == "Через пять минут по плану: «A» и «B»."


@pytest.mark.parametrize("n,text", [(1, "одну минуту"), (2, "две минуты"), (5, "пять минут"), (11, "одиннадцать минут"),
                                    (21, "двадцать одну минуту"), (34, "тридцать четыре минуты"), (60, "час"),
                                    (75, "час пятнадцать минут")])
def test_minutes_phrase(n, text):
    assert minutes_phrase(n) == text


def test_like_pattern_escapes_wildcards():
    assert _like_pattern("50%_x") == "%50\\%\\_x%"


# ---- MCP server ------------------------------------------------------------
class FakeDB:
    """Stands in for PlannerDB: records calls, has no write methods at all."""

    def __init__(self):
        self.calls = []
        self._user = {"id": 2, "name": "Vlad", "timezone": None}

    def user(self):
        return self._user

    def now(self):
        return dt.datetime(2026, 11, 10, 14, 18)

    def events(self, start, end):
        self.calls.append(("events", start, end))
        return [ev("Jarvis", 840, 887), ev("Все для мініка", 887, 932)]

    def todos(self, day=None):
        return [{"title": "купить кофе", "kind": "personal", "date": "2026-11-01"}]

    def tasks(self):
        return [{"title": "лендинг", "status": "doing", "priority": "P1", "due": "2026-11-09", "project": "ShareMe"}]

    def week_focus(self, day):
        return {"week_start": "2026-11-09", "focus": "запуск", "outcomes": []}

    def search(self, q, start, end):
        self.calls.append(("search", q, start, end))
        return []

    def close(self):
        pass


def rpc(method, params=None, mid=1, handlers=None):
    return handle_message({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}},
                          handlers or Handlers(FakeDB()))


def test_mcp_handshake_and_tools():
    init = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}})
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert init["result"]["capabilities"]["tools"]
    assert rpc("initialize", {"protocolVersion": "1999-01-01"})["result"]["protocolVersion"]  # falls back
    assert handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}, Handlers(None)) is None
    tools = rpc("tools/list")["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == {"planner_day", "planner_upcoming", "planner_range", "planner_search", "planner_todos",
                     "planner_tasks", "planner_week_focus"}
    assert all(t["annotations"]["readOnlyHint"] and not t["annotations"]["destructiveHint"] for t in tools)
    assert rpc("nope")["error"]["code"] == -32601


def test_mcp_has_no_write_tools():
    banned = ("add", "create", "update", "delete", "remove", "set", "move", "done", "write", "insert", "sql")
    for t in TOOLS:
        assert not any(b in t["name"].split("_") for b in banned), t["name"]


def test_mcp_tool_calls():
    h = Handlers(FakeDB())
    day = json.loads(rpc("tools/call", {"name": "planner_day", "arguments": {}}, handlers=h)["result"]["content"][0]["text"])
    assert day["date"] == "2026-11-10" and day["calendar"][0]["time"] == "14:00–14:47"
    assert "сейчас «Jarvis»" in day["now_line"]
    todos = json.loads(rpc("tools/call", {"name": "planner_todos"}, handlers=h)["result"]["content"][0]["text"])
    assert todos["todos"][0]["overdue"] is True
    up = json.loads(rpc("tools/call", {"name": "planner_upcoming", "arguments": {"hours": 3}}, handlers=h)
                    ["result"]["content"][0]["text"])
    assert up["now_running"]["title"] == "Jarvis" and [e["title"] for e in up["next"]] == ["Все для мініка"]
    bad = rpc("tools/call", {"name": "planner_day", "arguments": {"date": "когда-нибудь"}}, handlers=h)["result"]
    assert bad["isError"]
    unknown = rpc("tools/call", {"name": "planner_add_event", "arguments": {}}, handlers=h)
    assert unknown["error"]["code"] == -32602


def test_mcp_not_configured_reports_error():
    r = rpc("tools/call", {"name": "planner_day", "arguments": {}}, handlers=Handlers(None))["result"]
    assert r["isError"] and "PLANNER_DATABASE_URL" in r["content"][0]["text"]


def test_mcp_stdio_loop_is_ascii_json_lines():
    lines = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
             {"jsonrpc": "2.0", "method": "notifications/initialized"},
             {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "planner_day", "arguments": {}}}]
    stdin = io.BytesIO(("\n".join(json.dumps(m, ensure_ascii=False) for m in lines) + "\nnot json\n").encode("utf-8"))
    out = io.BytesIO()
    serve(stdin, out, Handlers(FakeDB()))
    resp = [json.loads(x) for x in out.getvalue().decode("ascii").splitlines()]
    assert [r.get("id") for r in resp] == [1, 2, None]
    assert "Jarvis" in resp[1]["result"]["content"][0]["text"]
    assert resp[2]["error"]["code"] == -32700


# ---- wiring into Claude and the persona ------------------------------------
def test_brain_command_gets_planner_mcp_only_when_configured(tmp_path):
    base = dict(model="sonnet", effort=None, system_prompt="x", cwd=tmp_path)
    cmd = build_command("claude", BrainSettings(**base))
    assert "--mcp-config" not in cmd and "--strict-mcp-config" in cmd
    servers = {"planner": {"type": "stdio", "command": "python", "args": ["-I", "planner_mcp.py"]}}
    cmd = build_command("claude", BrainSettings(**base, mcp_servers=servers, allowed_tools=["Read", "mcp__planner"]))
    cfg = json.loads(cmd[cmd.index("--mcp-config") + 1])
    assert cfg["mcpServers"]["planner"]["args"][0] == "-I"
    assert "mcp__planner" in cmd[cmd.index("--allowedTools") + 1]
    assert "postgres" not in " ".join(cmd)  # no secret on the command line


def test_persona_mentions_planner_only_when_enabled(tmp_path):
    on = persona.system_prompt("Джарвис", "сэр", tmp_path / "f.jpg", tmp_path, planner=True)
    off = persona.system_prompt("Джарвис", "сэр", tmp_path / "f.jpg", tmp_path)
    assert "mcp__planner" in on and "ТОЛЬКО ЧТЕНИЕ" in on and "mcp__planner" not in off
    assert "план: сейчас" in persona.turn_header(50, False, True, plan="план: сейчас «X» до 15:00")
    assert "план" not in persona.turn_header(50, False, True)


# ---- one-time setup ----------------------------------------------------------
def test_setup_sql_is_select_only():
    sql = " ".join(s for s, _ in setup_sql("pw'x"))
    assert "GRANT SELECT ON calendar_events, inbox_items, tasks, projects, week_plans, settings TO jarvis_ro" in sql
    assert "GRANT SELECT (id, name, email, timezone) ON users" in sql
    assert "default_transaction_read_only = on" in sql
    assert "PASSWORD 'pw''x'" in sql  # quote doubled
    for verb in ("INSERT", "UPDATE", "DELETE", "TRUNCATE", "ALL PRIVILEGES"):
        assert f"GRANT {verb}" not in sql


def test_readonly_url_and_env_write(tmp_path):
    url = readonly_url("postgresql://postgres:admin@x.proxy.rlwy.net:4242/railway", "a/b+c")
    assert url == "postgresql://jarvis_ro:a%2Fb%2Bc@x.proxy.rlwy.net:4242/railway?sslmode=require"
    assert "admin" not in url
    env = tmp_path / ".env"
    env.write_text("# keep\nOTHER=1\nPLANNER_USER=old\n", encoding="utf-8")
    write_env(env, {"PLANNER_DATABASE_URL": url, "PLANNER_USER": "Vlad"})
    assert read_env_file(env) == {"OTHER": "1", "PLANNER_USER": "Vlad", "PLANNER_DATABASE_URL": url}
    assert env.read_text(encoding="utf-8").startswith("# keep\n")


def test_env_is_git_ignored():
    from jarvis.config import ROOT

    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored


def test_pick_address_varies():
    from jarvis.planner import pick_address

    assert {pick_address("сэр", random.Random(i)) for i in range(30)} == {"сэр", "братан", "шеф"}


# ---- reminders inside the Jarvis core -------------------------------------
class _Voice:
    speaking = False
    last_sound_at = 0.0


def _jarvis_with_planner(tmp_path, now):
    import time

    from jarvis.config import Config
    from jarvis.core import Jarvis
    from jarvis.planner import PlannerService

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}", encoding="utf-8")
    j = Jarvis(Config(cfg_file))
    db = FakeDB()
    db.now = lambda: now
    svc = PlannerService(db)
    svc.events = [ev("Созвон", 900, 960)]  # 15:00
    svc.fetched_at = time.monotonic()
    j.planner, j.voice, j.window_open = svc, _Voice(), True
    return j


def test_core_speaks_reminder_once_when_quiet(tmp_path):
    j = _jarvis_with_planner(tmp_path, dt.datetime(2026, 11, 10, 14, 52))
    j._busy.set()  # Jarvis is answering: the reminder waits
    assert j._planner_tick() is None and j._turns.empty()
    j._busy.clear()
    text = j._planner_tick()
    assert "через восемь минут по плану: «Созвон»" in text
    turn = j._turns.get_nowait()
    assert turn.kind == "remind" and turn.text == text
    assert j._planner_tick() is None and j._turns.empty()  # never twice


def test_core_drops_reminder_missed_while_busy_and_no_window(tmp_path):
    j = _jarvis_with_planner(tmp_path, dt.datetime(2026, 11, 10, 14, 55))
    j._busy.set()
    j._planner_tick()
    j.planner.db.now = lambda: dt.datetime(2026, 11, 10, 15, 1)  # the call already started
    j._busy.clear()
    assert j._planner_tick() is None and j._turns.empty()
    j2 = _jarvis_with_planner(tmp_path, dt.datetime(2026, 11, 10, 14, 55))
    j2.window_open = False
    assert j2._planner_tick() is None


def test_core_reminders_off_in_config(tmp_path):
    j = _jarvis_with_planner(tmp_path, dt.datetime(2026, 11, 10, 14, 55))
    j.config._data["planner_reminders"] = []
    assert j._planner_tick() is None


def test_core_drops_waiting_reminder_when_block_done_meanwhile(tmp_path):
    j = _jarvis_with_planner(tmp_path, dt.datetime(2026, 11, 10, 14, 52))
    j._busy.set()
    j._planner_tick()
    assert j._remind_wait
    j.planner.events = [ev("Созвон", 900, 960, done=True)]
    j._busy.clear()
    assert j._planner_tick() is None and j._turns.empty()


def test_secret_lives_where_claude_may_not_read():
    from pathlib import Path

    from jarvis.config import ROOT, SECURITY_DENY
    from jarvis.planner import ENV_FILE

    assert ROOT not in ENV_FILE.parents  # not inside the repo
    assert str(ENV_FILE).lower().startswith(str(Path.home() / "AppData").lower())
    assert "Read(~/AppData/**)" in SECURITY_DENY


def test_workspace_timezone_fallback():
    from jarvis.planner import PlannerDB

    class DB(PlannerDB):
        def _query(self, sql, params=()):
            if "FROM users" in sql:
                return [(2, "Vlad", None)]
            assert "key = 'timezone'" in sql
            return [("Europe/Kyiv",)]

    db = DB(PlannerSettings("postgresql://x@h/db", "Vlad"))
    db.user()
    assert db.timezone_name() == "Europe/Kyiv"
    db2 = DB(PlannerSettings("postgresql://x@h/db", "Vlad", timezone="Asia/Makassar"))
    db2.user()
    assert db2.timezone_name() == "Asia/Makassar"
