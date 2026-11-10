"""Read-only access to the user's Planning System (its Postgres database).

Jarvis never writes there. Three layers keep it that way:
1. the database role in planner.env is a SELECT-only role (python -m jarvis.planner_setup);
2. every connection runs with `default_transaction_read_only=on` and every query in a
   READ ONLY transaction that is rolled back, never committed;
3. only the fixed, parameterised SELECTs below exist: Claude never sends SQL.

Connection settings live in %APPDATA%/Jarvis/planner.env — outside the repo and inside ~/AppData,
which Jarvis's Claude is never allowed to read (config.SECURITY_DENY):
    PLANNER_DATABASE_URL=postgresql://jarvis_ro:...@xxx.proxy.rlwy.net:12345/railway
    PLANNER_USER=Vlad            # name, email or numeric id in the planner's users table
    PLANNER_TIMEZONE=Europe/Kyiv # optional; default: the user's zone in the planner, else this PC's clock

Calendar model (Planning System, table calendar_events): `date` is the day column,
`start_min`/`end_min` are minutes from that day's midnight in the user's wall-clock time;
values above 1440 are hours after midnight that still belong to that column (a planner
day runs 01:00 -> 01:00).
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import os
import random
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit

from .config import ROOT

log = logging.getLogger("jarvis.planner")

def _secrets_dir() -> Path:
    base = os.environ.get("APPDATA")
    return (Path(base) if base else Path.home() / "AppData" / "Roaming") / "Jarvis"


ENV_FILE = _secrets_dir() / "planner.env"
DAY_SHIFT_MIN = 60  # the planner day starts at 01:00
MAX_RANGE_DAYS = 31
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}


class PlannerError(Exception):
    """A problem the user can fix (shown and spoken in Russian)."""


# ---- settings -------------------------------------------------------------
@dataclass(frozen=True)
class PlannerSettings:
    database_url: str
    user: str
    timezone: str = ""  # optional override (IANA name)

    def __repr__(self) -> str:  # never print the password
        return f"PlannerSettings(host={self.host!r}, user={self.user!r})"

    @property
    def host(self) -> str:
        try:
            return urlsplit(self.database_url).hostname or ""
        except ValueError:
            return ""


def read_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser: comments, blank lines, optional quotes, `export ` prefix, BOM."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return {}
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.lower().startswith("export "):
            key = key[7:].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key] = value
    return out


def load_settings(path: Path = ENV_FILE) -> Optional[PlannerSettings]:
    """None when the planner is not configured (no PLANNER_DATABASE_URL)."""
    values = read_env_file(path)
    url = values.get("PLANNER_DATABASE_URL", "").strip()
    if not url:
        return None
    if not re.match(r"^postgres(ql)?://", url):
        raise PlannerError("PLANNER_DATABASE_URL в planner.env должен начинаться с postgresql://")
    return PlannerSettings(database_url=url, user=values.get("PLANNER_USER", "").strip(),
                           timezone=values.get("PLANNER_TIMEZONE", "").strip())


# ---- pure helpers (tested without a database) -----------------------------
@dataclass(frozen=True)
class Event:
    id: str
    title: str
    date: dt.date
    start_min: int
    end_min: int
    done: bool

    @property
    def start(self) -> dt.datetime:
        return dt.datetime.combine(self.date, dt.time()) + dt.timedelta(minutes=self.start_min)

    @property
    def end(self) -> dt.datetime:
        return dt.datetime.combine(self.date, dt.time()) + dt.timedelta(minutes=max(self.end_min, self.start_min))

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "time": f"{hhmm(self.start_min)}–{hhmm(self.end_min)}",
            "title": self.title,
            "done": self.done,
            **({"after_midnight": True} if self.start_min >= 1440 else {}),
        }


def hhmm(minutes: int) -> str:
    minutes = int(minutes) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def planner_day(now: dt.datetime) -> dt.date:
    """The column of the weekly planner that `now` belongs to (00:30 is still yesterday)."""
    return (now - dt.timedelta(minutes=DAY_SHIFT_MIN)).date()


def parse_day(value: Optional[str], now: dt.datetime) -> dt.date:
    """YYYY-MM-DD, or today / tomorrow / yesterday (also in Russian)."""
    today = planner_day(now)
    v = (value or "").strip().lower()
    if v in ("", "today", "сегодня"):
        return today
    if v in ("tomorrow", "завтра"):
        return today + dt.timedelta(days=1)
    if v in ("yesterday", "вчера"):
        return today - dt.timedelta(days=1)
    if v in ("послезавтра", "day after tomorrow"):
        return today + dt.timedelta(days=2)
    try:
        return dt.date.fromisoformat(v)
    except ValueError:
        raise PlannerError(f"Дата должна быть в формате ГГГГ-ММ-ДД, получено: {value!r}") from None


def now_and_next(events: Iterable[Event], now: dt.datetime) -> tuple[Optional[Event], list[Event]]:
    """The event running now (not done) and the following ones today (planner day)."""
    today = planner_day(now)
    current: Optional[Event] = None
    upcoming: list[Event] = []
    for ev in sorted(events, key=lambda e: e.start):
        if ev.done or ev.date != today:
            continue
        if ev.start <= now < ev.end and current is None:
            current = ev
        elif ev.start > now:
            upcoming.append(ev)
    return current, upcoming


def header_line(events: Iterable[Event], now: dt.datetime, limit: int = 2) -> str:
    """One short line for the per-turn context header: what is on now and next."""
    current, upcoming = now_and_next(events, now)
    parts = []
    if current is not None:
        parts.append(f"сейчас «{_short(current.title)}» до {hhmm(current.end_min)}")
    if upcoming:
        nxt = "; ".join(f"{hhmm(e.start_min)} «{_short(e.title)}»" for e in upcoming[:limit])
        parts.append(f"дальше {nxt}")
    if not parts:
        return "план на сегодня: больше ничего не запланировано"
    return "план: " + "; ".join(parts)


def _short(title: str, limit: int = 60) -> str:
    title = " ".join(str(title).split())
    return title if len(title) <= limit else title[: limit - 1] + "…"


@dataclass(frozen=True)
class Reminder:
    key: tuple
    event: Event
    lead: int  # minutes before the start this reminder is for (0 = at the start)


START_GRACE_MIN = 3  # a "starts now" reminder still makes sense this long after the start


def due_reminders(events: Iterable[Event], now: dt.datetime, leads: Iterable[int], sent: set) -> list[Reminder]:
    """Reminders that should be spoken now and were not spoken yet.

    A lead-N reminder is due from N minutes before the start until the start (so one
    missed while Jarvis was busy or just starting still comes, with the real time left);
    a lead-0 reminder is due for a few minutes after the start. Done events are skipped.
    """
    leads = sorted({max(0, int(x)) for x in leads}, reverse=True)
    out: list[Reminder] = []
    for ev in events:
        if ev.done:
            continue
        start = ev.start
        for i, lead in enumerate(leads):
            key = (ev.id, start.isoformat(), lead)
            if key in sent:
                continue
            if lead > 0:
                # Only until the next, closer reminder takes over (e.g. 10 -> 0).
                closer = leads[i + 1] if i + 1 < len(leads) else None
                until = start - dt.timedelta(minutes=closer) if closer else start
                if start - dt.timedelta(minutes=lead) <= now < until:
                    out.append(Reminder(key, ev, lead))
            elif start <= now < start + dt.timedelta(minutes=START_GRACE_MIN):
                out.append(Reminder(key, ev, lead))
    return out


def still_due(r: Reminder, now: dt.datetime) -> bool:
    if r.lead > 0:
        return now < r.event.start
    return now < r.event.start + dt.timedelta(minutes=START_GRACE_MIN)


_UNITS = ["", "одну", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять",
          "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
          "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = {20: "двадцать", 30: "тридцать", 40: "сорок", 50: "пятьдесят"}


def minutes_phrase(n: int) -> str:
    """'через …' object in words, as the speech synthesiser should say it: «одну минуту», «пять минут»."""
    n = max(1, int(n))
    if n >= 60:
        hours, rest = divmod(n, 60)
        hw = {1: "час", 2: "два часа", 3: "три часа", 4: "четыре часа"}.get(hours, f"{hours} часов")
        return hw if rest == 0 else f"{hw} {minutes_phrase(rest)}"
    if n < 20:
        words = _UNITS[n]
    else:
        tens, unit = divmod(n, 10)
        words = _TENS[tens * 10] + (f" {_UNITS[unit]}" if unit else "")
    last2, last = n % 100, n % 10
    if 11 <= last2 <= 14:
        noun = "минут"
    elif last == 1:
        noun = "минуту"
    elif 2 <= last <= 4:
        noun = "минуты"
    else:
        noun = "минут"
    return f"{words} {noun}"


def reminder_text(reminders: list[Reminder], now: dt.datetime, address: str) -> str:
    """One spoken sentence for reminders that fire together."""
    titles = []
    for r in sorted(reminders, key=lambda r: r.event.start):
        t = _short(r.event.title, 80)
        if t not in titles:
            titles.append(t)
    what = " и ".join(f"«{t}»" for t in titles)
    first = min(r.event.start for r in reminders)
    left = math.ceil((first - now).total_seconds() / 60)
    address = address[:1].upper() + address[1:] if address else ""
    lead = f"{address}, " if address else ""
    if left <= 0:
        text = f"{lead}по плану сейчас: {what}."
    else:
        text = f"{lead}через {minutes_phrase(left)} по плану: {what}."
    return text[:1].upper() + text[1:]


def pick_address(user_title: str, rng=random) -> str:
    return rng.choice([user_title or "сэр", "братан", "шеф"])


def _like_pattern(query: str) -> str:
    q = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{q}%"


# ---- database access ------------------------------------------------------
class PlannerDB:
    """A small read-only client. Thread-safe; one lazily (re)opened connection."""

    def __init__(self, settings: PlannerSettings, connect_timeout: int = 8, statement_timeout_ms: int = 8000):
        self.settings = settings
        self.connect_timeout = connect_timeout
        self.statement_timeout_ms = statement_timeout_ms
        self._conn = None
        self._lock = threading.Lock()
        self._user: Optional[dict] = None

    # connection ---------------------------------------------------------
    def _connect(self):
        import psycopg

        kwargs: dict[str, Any] = {
            "connect_timeout": self.connect_timeout,
            "autocommit": False,
            "application_name": "jarvis-readonly",
            "prepare_threshold": None,  # no server-side prepared statements: safe behind any pooler
            "options": f"-c default_transaction_read_only=on -c statement_timeout={int(self.statement_timeout_ms)}",
        }
        # Encrypt anything that leaves this machine unless the URL says otherwise.
        if "sslmode=" not in self.settings.database_url and self.settings.host not in _LOCAL_HOSTS:
            kwargs["sslmode"] = "require"
        conn = psycopg.connect(self.settings.database_url, **kwargs)
        conn.read_only = True  # every transaction: BEGIN READ ONLY
        return conn

    def close(self) -> None:
        with self._lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _query(self, sql: str, params: tuple = ()) -> list[tuple]:
        import psycopg

        with self._lock:
            for attempt in (1, 2):
                if self._conn is None or self._conn.closed:
                    try:
                        self._conn = self._connect()
                    except psycopg.Error as exc:
                        raise PlannerError(f"Не могу подключиться к базе планера ({_reason(exc)}).") from exc
                conn = self._conn
                try:
                    with conn.cursor() as cur:
                        cur.execute(sql, params)
                        rows = cur.fetchall()
                    conn.rollback()  # read-only: nothing is ever committed
                    return rows
                except psycopg.errors.QueryCanceled as exc:  # statement_timeout: do not retry
                    try:
                        conn.rollback()
                    except Exception:
                        self._conn = None
                    raise PlannerError("Планер слишком долго отвечает, попробуйте позже.") from exc
                except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
                    # A dropped connection (sleep, Wi-Fi change): reconnect once.
                    self._conn = None
                    try:
                        conn.close()
                    except Exception:
                        pass
                    if attempt == 2:
                        raise PlannerError(f"База планера недоступна ({_reason(exc)}).") from exc
                except psycopg.errors.InsufficientPrivilege as exc:
                    conn.rollback()
                    raise PlannerError("У пользователя базы нет прав на чтение этой таблицы "
                                       "(выполните docs/planner-readonly.sql).") from exc
                except psycopg.Error as exc:
                    try:
                        conn.rollback()
                    except Exception:
                        self._conn = None
                    raise PlannerError(f"Ошибка запроса к планеру ({_reason(exc)}).") from exc
            raise PlannerError("База планера недоступна.")

    # identity -----------------------------------------------------------
    def user(self) -> dict:
        """The planner user Jarvis works for: {id, name, timezone}. Cached."""
        if self._user is not None:
            return self._user
        ref = self.settings.user
        if not ref:
            raise PlannerError("В planner.env не указан PLANNER_USER (имя, email или id пользователя в планере).")
        rows = self._query(
            "SELECT id, name, timezone FROM users "
            "WHERE lower(name) = lower(%s) OR lower(email) = lower(%s) OR id::text = %s ORDER BY id LIMIT 3",
            (ref, ref, ref),
        )
        if not rows:
            raise PlannerError(f"Пользователь {ref!r} не найден в планере (PLANNER_USER в planner.env).")
        if len(rows) > 1:
            raise PlannerError(f"PLANNER_USER={ref!r} подходит нескольким пользователям, укажите email или id.")
        uid, name, tz = rows[0]
        user = {"id": int(uid), "name": name, "timezone": tz, "workspace_timezone": None}
        if not tz and not self.settings.timezone:
            # Like the platform: no personal zone -> the workspace zone (settings.timezone, default Kyiv).
            try:
                ws = self._query("SELECT value FROM settings WHERE key = 'timezone'")
                value = ws[0][0] if ws else None
                user["workspace_timezone"] = value if isinstance(value, str) and value else "Europe/Kyiv"
            except PlannerError as exc:
                log.info("workspace time zone unavailable (%s), using this PC's clock", exc)
        self._user = user
        return self._user

    def now(self) -> dt.datetime:
        """Wall-clock time the planner shows for this user (naive), like the platform's now-line:
        PLANNER_TIMEZONE, else the user's zone in the planner, else the workspace zone, else this PC's clock."""
        tz = self.timezone_name()
        if tz:
            try:
                from zoneinfo import ZoneInfo

                return dt.datetime.now(ZoneInfo(tz)).replace(tzinfo=None, microsecond=0)
            except Exception:
                log.debug("unknown time zone %r, using local time", tz)
        return dt.datetime.now().replace(microsecond=0)

    def timezone_name(self) -> Optional[str]:
        u = self._user or {}
        return self.settings.timezone or u.get("timezone") or u.get("workspace_timezone") or None

    # reads ----------------------------------------------------------------
    def events(self, start: dt.date, end: dt.date) -> list[Event]:
        if end < start:
            start, end = end, start
        if (end - start).days > MAX_RANGE_DAYS:
            raise PlannerError(f"Слишком длинный период: не больше {MAX_RANGE_DAYS} дней за раз.")
        uid = self.user()["id"]
        rows = self._query(
            "SELECT id, title, date, start_min, end_min, done FROM calendar_events "
            "WHERE user_id = %s AND date BETWEEN %s AND %s ORDER BY date, start_min LIMIT 1000",
            (uid, start, end),
        )
        return [Event(str(r[0]), r[1], _as_date(r[2]), int(r[3]), int(r[4]), bool(r[5])) for r in rows]

    def todos(self, day: Optional[dt.date] = None) -> list[dict]:
        """Open to-dos (Tasks page, under the planner). With `day`: dated that day + undated."""
        uid = self.user()["id"]
        rows = self._query(
            "SELECT title, kind, date FROM inbox_items WHERE user_id = %s AND NOT done "
            "ORDER BY date NULLS LAST, created_at LIMIT 300",
            (uid,),
        )
        out = []
        for title, kind, date in rows:
            d = _as_date(date) if date is not None else None
            if day is not None and d is not None and d != day:
                continue
            out.append({"title": title, "kind": kind, "date": d.isoformat() if d else None})
        return out

    def tasks(self) -> list[dict]:
        """Open team tasks (kanban) assigned to the user."""
        uid = self.user()["id"]
        rows = self._query(
            "SELECT t.title, t.status, t.priority, t.due_date, p.name FROM tasks t "
            "LEFT JOIN projects p ON p.id = t.project_id "
            "WHERE t.assignee_id = %s AND t.status <> 'done' AND NOT t.is_demo "
            "ORDER BY t.due_date NULLS LAST, t.priority, t.id LIMIT 200",
            (uid,),
        )
        return [{"title": r[0], "status": r[1], "priority": f"P{r[2]}",
                 "due": _as_date(r[3]).isoformat() if r[3] is not None else None, "project": r[4]} for r in rows]

    def week_focus(self, day: dt.date) -> Optional[dict]:
        monday = day - dt.timedelta(days=day.weekday())
        uid = self.user()["id"]
        rows = self._query(
            "SELECT focus, items FROM week_plans WHERE week_start = %s AND (user_id = %s OR user_id IS NULL) "
            "AND NOT is_demo ORDER BY user_id NULLS LAST LIMIT 1",
            (monday, uid),
        )
        if not rows:
            return None
        focus, items = rows[0]
        outcomes = [{"text": i.get("text"), "done": i.get("done")} for i in (items or []) if isinstance(i, dict)]
        return {"week_start": monday.isoformat(), "focus": focus, "outcomes": outcomes}

    def search(self, query: str, start: dt.date, end: dt.date) -> list[Event]:
        query = (query or "").strip()
        if not query:
            raise PlannerError("Пустой запрос для поиска.")
        uid = self.user()["id"]
        rows = self._query(
            "SELECT id, title, date, start_min, end_min, done FROM calendar_events "
            "WHERE user_id = %s AND date BETWEEN %s AND %s AND title ILIKE %s "
            "ORDER BY date, start_min LIMIT 50",
            (uid, start, end, _like_pattern(query[:100])),
        )
        return [Event(str(r[0]), r[1], _as_date(r[2]), int(r[3]), int(r[4]), bool(r[5])) for r in rows]


def _as_date(value: Any) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _reason(exc: BaseException) -> str:
    """A short, password-free reason for logs and the user."""
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    text = re.sub(r"(?i)password\S*", "password=***", text)
    text = re.sub(r"\"[^\"]*\"|'[^']*'", '"…"', text)  # quoted fragments may hold part of a secret
    text = re.sub(r"postgres(ql)?://\S+", "postgresql://***", text)
    return f"{type(exc).__name__}: {text[:160]}"


# ---- the background service used by the Jarvis core -----------------------
class PlannerService:
    """Keeps a fresh copy of the near calendar for reminders and the context header."""

    def __init__(self, db: PlannerDB, poll_seconds: float = 60.0):
        self.db = db
        self.poll_seconds = max(15.0, float(poll_seconds))
        self.events: list[Event] = []
        self.fetched_at = 0.0  # monotonic time of the last successful fetch
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    def refresh(self, clock=None) -> bool:
        import time

        try:
            now = self.db.now()
            if self.db._user is None:
                self.db.user()
                now = self.db.now()  # now in the user's time zone
            today = planner_day(now)
            events = self.db.events(today - dt.timedelta(days=1), today + dt.timedelta(days=1))
        except PlannerError as exc:
            with self._lock:
                changed = self.error != str(exc)
                self.error = str(exc)
            if changed:
                log.warning("planner: %s", exc)
            return False
        except Exception as exc:  # never let the reminder thread die
            with self._lock:
                self.error = f"Ошибка планера ({_reason(exc)})."
            log.exception("planner refresh failed")
            return False
        with self._lock:
            if self.error:
                log.info("planner: connected again")
            self.events, self.error = events, None
            self.fetched_at = (clock or time.monotonic)()
        return True

    def fresh(self, max_age: float = 300.0) -> bool:
        import time

        return self.fetched_at > 0 and time.monotonic() - self.fetched_at < max_age

    def snapshot(self) -> list[Event]:
        with self._lock:
            return list(self.events)

    def status_text(self) -> str:
        if self.error:
            return "ошибка"
        user = self.db._user
        if user is None:
            return "подключение"
        return f"Planning System · {user['name']}"
