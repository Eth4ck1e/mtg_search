"""Apply pending SQL migrations.

Reads .sql files from src/db/migrations/ in lexicographic order, executes
each inside a transaction, and records the filename in schema_migrations.
Idempotent — already-applied migrations are skipped.

Usage:
    python scripts/migrate.py            # apply pending
    python scripts/migrate.py --status   # show applied vs pending, no changes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg

from src.config import settings

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "src" / "db" / "migrations"

CREATE_BOOKKEEPING = """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version    TEXT        PRIMARY KEY,
        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
"""


def discover_migrations() -> list[Path]:
    if not MIGRATIONS_DIR.exists():
        return []
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def fetch_applied(conn: psycopg.Connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(CREATE_BOOKKEEPING)
        conn.commit()
        cur.execute("SELECT version FROM schema_migrations")
        return {row[0] for row in cur.fetchall()}


def apply_one(conn: psycopg.Connection, sql_file: Path) -> None:
    sql = sql_file.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
        cur.execute(
            "INSERT INTO schema_migrations (version) VALUES (%s)",
            (sql_file.name,),
        )
    conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show applied and pending migrations without changing anything.",
    )
    args = parser.parse_args()

    all_migrations = discover_migrations()
    if not all_migrations:
        print(f"No migration files found under {MIGRATIONS_DIR}.")
        return 0

    try:
        conn = psycopg.connect(settings.database_url, autocommit=False)
    except psycopg.OperationalError as exc:
        print(f"Could not connect to {settings.database_url}: {exc}", file=sys.stderr)
        print("Is the postgres container running? (`docker compose up -d`)", file=sys.stderr)
        return 1

    try:
        applied = fetch_applied(conn)
        pending = [m for m in all_migrations if m.name not in applied]

        if args.status:
            print(f"Applied  ({len(applied)}):")
            for name in sorted(applied):
                print(f"  - {name}")
            print(f"Pending  ({len(pending)}):")
            for m in pending:
                print(f"  - {m.name}")
            return 0

        if not pending:
            print(f"No migrations pending. ({len(applied)} already applied.)")
            return 0

        for sql_file in pending:
            print(f"Applying {sql_file.name}...", flush=True)
            apply_one(conn, sql_file)
            print("  ok")

        print(f"Applied {len(pending)} migration(s).")
        return 0
    except Exception as exc:
        conn.rollback()
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
