"""Database schema and idempotent migration runner for StrikeShield."""

import sqlite3
from pathlib import Path
from typing import Optional
from strikeshield.config import settings

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Returns an active SQLite connection configured with WAL and foreign keys."""
    target_path = db_path or settings.database_path
    conn = sqlite3.connect(str(target_path))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Ensures database directory exists and creates schema_version table."""
    target_path = db_path or settings.database_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(target_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()


def get_current_schema_version(conn: sqlite3.Connection) -> int:
    """Queries the latest applied schema version."""
    cursor = conn.execute("SELECT MAX(version) FROM schema_version;")
    row = cursor.fetchone()
    if row and row[0] is not None:
        return int(row[0])
    return 0


def migrate(db_path: Optional[Path] = None) -> int:
    """Applies pending SQL migrations in numerical order. Idempotent."""
    init_db(db_path)
    target_path = db_path or settings.database_path

    with get_connection(target_path) as conn:
        current_version = get_current_schema_version(conn)

        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        applied_any = False

        for mig in migration_files:
            try:
                version_num = int(mig.name.split("_")[0])
            except (ValueError, IndexError):
                continue

            if version_num > current_version:
                with open(mig, "r", encoding="utf-8") as f:
                    sql_script = f.read()

                conn.executescript(sql_script)
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?);",
                    (version_num,),
                )
                conn.commit()
                applied_any = True
                current_version = version_num

        return current_version
