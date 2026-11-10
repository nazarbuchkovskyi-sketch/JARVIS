-- Manual alternative to `python -m jarvis.planner_setup`: a SELECT-only role for Jarvis.
-- Run once as the database owner (e.g. psql "$DATABASE_PUBLIC_URL"). Replace the password.
DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'jarvis_ro') THEN CREATE ROLE jarvis_ro LOGIN; END IF; END $$;
ALTER ROLE jarvis_ro WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
  CONNECTION LIMIT 5 PASSWORD 'CHANGE_ME_long_random_password';
ALTER ROLE jarvis_ro SET default_transaction_read_only = on;
ALTER ROLE jarvis_ro SET statement_timeout = '10s';
DO $$ BEGIN EXECUTE format('GRANT CONNECT ON DATABASE %I TO jarvis_ro', current_database()); END $$;
GRANT USAGE ON SCHEMA public TO jarvis_ro;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM jarvis_ro;
GRANT SELECT ON calendar_events, inbox_items, tasks, projects, week_plans, settings TO jarvis_ro;  -- settings: only the workspace timezone is read
GRANT SELECT (id, name, email, timezone) ON users TO jarvis_ro;  -- never password_hash
-- Then in %APPDATA%\Jarvis\planner.env:
--   PLANNER_DATABASE_URL=postgresql://jarvis_ro:<password>@<host>.proxy.rlwy.net:<port>/railway?sslmode=require
--   PLANNER_USER=Vlad
-- To revoke later:  DROP OWNED BY jarvis_ro; DROP ROLE jarvis_ro;
