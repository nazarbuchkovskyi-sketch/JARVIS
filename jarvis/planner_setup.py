"""One-time setup of Jarvis's read-only access to the Planning System database.

    python -m jarvis.planner_setup            # create the read-only role and write planner.env
    python -m jarvis.planner_setup --check    # test the saved connection and print today's plan

The setup asks for the database OWNER connection string (Railway: Postgres -> Variables ->
DATABASE_PUBLIC_URL) with hidden input, uses it once and never stores it. It creates (or
re-keys) a login role `jarvis_ro` that can only SELECT the few tables Jarvis needs, checks that
the role really cannot write, and saves the read-only connection string to %APPDATA%/Jarvis/planner.env
(outside the repo; Jarvis's Claude may not read ~/AppData).
"""
from __future__ import annotations

import argparse
import getpass
import os
import secrets
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit

from .planner import ENV_FILE, PlannerDB, PlannerError, header_line, load_settings, planner_day, read_env_file

ROLE = "jarvis_ro"
READ_TABLES = ("calendar_events", "inbox_items", "tasks", "projects", "week_plans", "settings")
USER_COLUMNS = ("id", "name", "email", "timezone")  # never password_hash
ALL_CHECKED = READ_TABLES + ("users",)


def setup_sql(password: str) -> list[tuple[str, tuple]]:
    """Idempotent statements run as the database owner (the password is a bound parameter where possible)."""
    pw_literal = "'" + password.replace("'", "''") + "'"  # CREATE/ALTER ROLE cannot take a bind parameter
    tables = ", ".join(READ_TABLES)
    cols = ", ".join(USER_COLUMNS)
    return [
        (f"DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{ROLE}') THEN "
         f"CREATE ROLE {ROLE} LOGIN; END IF; END $$", ()),
        (f"ALTER ROLE {ROLE} WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS "
         f"CONNECTION LIMIT 5 PASSWORD {pw_literal}", ()),
        (f"ALTER ROLE {ROLE} SET default_transaction_read_only = on", ()),
        (f"ALTER ROLE {ROLE} SET statement_timeout = '10s'", ()),
        (f"DO $$ BEGIN EXECUTE format('GRANT CONNECT ON DATABASE %I TO {ROLE}', current_database()); END $$", ()),
        (f"GRANT USAGE ON SCHEMA public TO {ROLE}", ()),
        (f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {ROLE}", ()),
        (f"GRANT SELECT ON {tables} TO {ROLE}", ()),
        (f"GRANT SELECT ({cols}) ON users TO {ROLE}", ()),
    ]


def readonly_url(admin_url: str, password: str) -> str:
    p = urlsplit(admin_url)
    if not p.hostname:
        raise PlannerError("В строке подключения нет адреса сервера.")
    host = f"[{p.hostname}]" if ":" in p.hostname else p.hostname
    port = f":{p.port}" if p.port else ""
    db = p.path or "/postgres"
    ssl = "" if p.hostname in ("localhost", "127.0.0.1", "::1") else "?sslmode=require"
    return f"postgresql://{ROLE}:{quote(password, safe='')}@{host}{port}{db}{ssl}"


def write_env(path: Path, values: dict[str, str]) -> None:
    """Set keys in the env file, keeping every other line as it was."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.exists() else []
    done = set()
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key in values:
            out.append(f"{key}={values[key]}")
            done.add(key)
        else:
            out.append(line)
    for key, value in values.items():
        if key not in done:
            out.append(f"{key}={value}")
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    tmp.replace(path)


def _connect_admin(url: str):
    import psycopg

    kwargs = {"connect_timeout": 10, "autocommit": True}
    host = urlsplit(url).hostname or ""
    if "sslmode=" not in url and host not in ("localhost", "127.0.0.1", "::1"):
        kwargs["sslmode"] = "require"
    return psycopg.connect(url, **kwargs)


def verify_readonly(url: str) -> list[str]:
    """Problems found with the read-only role (empty list = fine). Uses only privilege checks, never writes."""
    import psycopg

    problems = []
    kwargs = {"connect_timeout": 10}
    host = urlsplit(url).hostname or ""
    if "sslmode=" not in url and host not in ("localhost", "127.0.0.1", "::1"):
        kwargs["sslmode"] = "require"
    with psycopg.connect(url, **kwargs) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("SHOW default_transaction_read_only")
            if cur.fetchone()[0] != "on":
                problems.append("default_transaction_read_only выключен")
            cur.execute("SELECT rolsuper OR rolcreaterole OR rolcreatedb FROM pg_roles WHERE rolname = current_user")
            if cur.fetchone()[0]:
                problems.append("у роли есть административные права")
            for table in ALL_CHECKED:
                cur.execute("SELECT has_table_privilege(current_user, %s, 'INSERT') OR "
                            "has_table_privilege(current_user, %s, 'UPDATE') OR "
                            "has_table_privilege(current_user, %s, 'DELETE') OR "
                            "has_table_privilege(current_user, %s, 'TRUNCATE')", (table,) * 4)
                if cur.fetchone()[0]:
                    problems.append(f"роль может изменять таблицу {table}")
            cur.execute("SELECT has_column_privilege(current_user, 'users', 'password_hash', 'SELECT')")
            if cur.fetchone()[0]:
                problems.append("роль может читать users.password_hash")
        conn.rollback()
    return problems


def run_setup(admin_url: str, user_ref: str, env_path: Path = ENV_FILE) -> str:
    password = secrets.token_urlsafe(32)
    with _connect_admin(admin_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.calendar_events') IS NOT NULL")
        if not cur.fetchone()[0]:
            raise PlannerError("В этой базе нет таблицы calendar_events — это точно база Planning System?")
        cur.execute("SELECT id, name, email FROM users ORDER BY id")
        people = cur.fetchall()
        # Check the user BEFORE touching the role: a typo must not re-key a working setup.
        match = [p for p in people if user_ref.lower() in (str(p[0]), (p[1] or "").lower(), (p[2] or "").lower())]
        if len(match) != 1:
            names = ", ".join(f"{p[1]} (id {p[0]})" for p in people)
            raise PlannerError(f"Пользователь {user_ref!r} не найден однозначно. Есть: {names}")
        for sql, params in setup_sql(password):
            cur.execute(sql, params or None)  # no params: % in the SQL is literal
    url = readonly_url(admin_url, password)
    problems = verify_readonly(url)
    if problems:
        raise PlannerError("Роль создана, но проверка не пройдена: " + "; ".join(problems))
    write_env(env_path, {"PLANNER_DATABASE_URL": url, "PLANNER_USER": match[0][1]})
    return match[0][1]


def run_check(env_path: Path = ENV_FILE) -> int:
    settings = load_settings(env_path)
    if settings is None:
        print(f"В {env_path} нет PLANNER_DATABASE_URL. Сначала: python -m jarvis.planner_setup")
        return 1
    db = PlannerDB(settings)
    try:
        user = db.user()
        now = db.now()
        today = planner_day(now)
        events = db.events(today, today)
        problems = verify_readonly(settings.database_url)
    finally:
        db.close()
    print(f"Подключено: {settings.host}, пользователь планера: {user['name']} (часовой пояс {db.timezone_name() or 'как у этого ПК'})")
    print(f"Сейчас {now:%Y-%m-%d %H:%M}, день планера {today}: блоков {len(events)}")
    for e in events:
        d = e.as_dict()
        print(f"  {d['time']}  {'[x]' if e.done else '[ ]'} {e.title}")
    print(header_line(events, now))
    if problems:
        print("ВНИМАНИЕ, роль может больше, чем читать: " + "; ".join(problems))
        return 2
    print("Доступ только на чтение: OK")
    return 0


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="проверить сохранённое подключение")
    ap.add_argument("--user", help="имя, email или id пользователя в планере (например Vlad)")
    args = ap.parse_args(argv)
    try:
        if args.check:
            return run_check()
        admin = os.environ.get("PLANNER_ADMIN_URL") or getpass.getpass(
            "Вставьте DATABASE_PUBLIC_URL из Railway (ввод скрыт) и нажмите Enter: ").strip()
        if not admin.startswith(("postgres://", "postgresql://")):
            print("Это не похоже на строку подключения postgresql://…")
            return 1
        if "railway.internal" in admin:
            print("Это внутренний адрес Railway (DATABASE_URL). Нужен DATABASE_PUBLIC_URL "
                  "(Postgres → Settings → Networking → Public Networking / TCP Proxy).")
            return 1
        user_ref = args.user or input("Ваше имя в планере (например Vlad): ").strip()
        name = run_setup(admin, user_ref)
        print(f"Готово: роль {ROLE} (только чтение) создана, подключение сохранено в {ENV_FILE}, пользователь планера: {name}.")
        print("Строку администратора нигде не сохраняю. Проверка: python -m jarvis.planner_setup --check")
        return 0
    except PlannerError as exc:
        print(f"Ошибка: {exc}")
        return 1
    except Exception as exc:
        from .planner import _reason

        print(f"Ошибка: {_reason(exc)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
