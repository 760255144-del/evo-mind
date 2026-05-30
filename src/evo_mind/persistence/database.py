"""aiosqlite connection pool and database lifecycle management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class Database:
    """Async SQLite database with connection pool and migration support."""

    def __init__(self, path: Path | str, pool_size: int = 5, busy_timeout_ms: int = 5000) -> None:
        self.path = Path(path) if isinstance(path, str) else path
        self.pool_size = pool_size
        self.busy_timeout_ms = busy_timeout_ms
        self._write_conn: aiosqlite.Connection | None = None
        self._read_conns: list[aiosqlite.Connection] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Set up the database directory and apply migrations."""
        if self._initialized:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_conn = await self._create_connection()
        await self._enable_wal()
        await self._apply_migrations()
        self._initialized = True
        logger.info("database_initialized", path=str(self.path))

    async def _create_connection(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(
            str(self.path),
            timeout=self.busy_timeout_ms / 1000.0,
        )
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return conn

    async def _enable_wal(self) -> None:
        assert self._write_conn is not None
        await self._write_conn.execute("PRAGMA journal_mode=WAL")

    async def _apply_migrations(self) -> None:
        """Apply pending schema migrations from the migrations/ directory."""
        assert self._write_conn is not None

        # Ensure schema_version table exists
        await self._write_conn.execute(
            """CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT (datetime('now')),
                description TEXT NOT NULL
            )"""
        )

        # Get current version
        cursor = await self._write_conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version"
        )
        row = await cursor.fetchone()
        current_version: int = row[0] if row else 0

        # Find and apply pending migrations
        migrations_dir = Path(__file__).parent.parent.parent.parent / "migrations"
        if not migrations_dir.exists():
            await self._write_conn.commit()
            return

        for migration_file in sorted(migrations_dir.glob("*.sql")):
            # Parse version number from filename (e.g., "001_initial_schema.sql")
            try:
                version = int(migration_file.stem.split("_")[0])
            except (ValueError, IndexError):
                continue

            if version <= current_version:
                continue

            logger.info("applying_migration", version=version, file=migration_file.name)
            sql = migration_file.read_text()
            await self._write_conn.executescript(sql)
            await self._write_conn.execute(
                "INSERT OR REPLACE INTO schema_version (version, description) VALUES (?, ?)",
                (version, migration_file.stem),
            )

        await self._write_conn.commit()

    async def execute(self, sql: str, parameters: Any = None) -> aiosqlite.Cursor:
        """Execute a write query on the write connection."""
        assert self._write_conn is not None, "Database not initialized"
        return await self._write_conn.execute(sql, parameters or [])

    async def execute_many(self, sql: str, parameters: list[Any]) -> aiosqlite.Cursor:
        """Execute a batch write query."""
        assert self._write_conn is not None, "Database not initialized"
        return await self._write_conn.executemany(sql, parameters)

    async def fetch_one(self, sql: str, parameters: Any = None) -> aiosqlite.Row | None:
        """Execute a read query returning one row or None."""
        assert self._write_conn is not None, "Database not initialized"
        cursor = await self._write_conn.execute(sql, parameters or [])
        return await cursor.fetchone()

    async def fetch_all(self, sql: str, parameters: Any = None) -> list[aiosqlite.Row]:
        """Execute a read query returning all rows."""
        assert self._write_conn is not None, "Database not initialized"
        cursor = await self._write_conn.execute(sql, parameters or [])
        return await cursor.fetchall()

    async def commit(self) -> None:
        """Commit pending write transaction."""
        assert self._write_conn is not None, "Database not initialized"
        await self._write_conn.commit()

    async def close(self) -> None:
        """Close all connections."""
        if self._write_conn:
            await self._write_conn.close()
            self._write_conn = None
        for conn in self._read_conns:
            await conn.close()
        self._read_conns.clear()
        self._initialized = False
        logger.info("database_closed", path=str(self.path))
