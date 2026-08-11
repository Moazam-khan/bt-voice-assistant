"""Structured conversation history for BT, using SQLite.

sqlite3 is synchronous, so every call runs via asyncio.to_thread to keep
the event loop free, per this project's async I/O rule.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from bt_core.logging_setup import get_logger

log = get_logger(__name__)


class ConversationTurn(BaseModel):
    """One recorded turn in BT's conversation history."""

    role: str
    content: str
    timestamp: datetime


class ConversationStore:
    """SQLite-backed store of BT's conversation history."""

    def __init__(self, db_path: Path) -> None:
        """Initialize the store and create its table if needed.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create the conversation_turns table if it doesn't already exist."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )

    async def add_turn(self, turn: ConversationTurn) -> None:
        """Record one conversation turn.

        Args:
            turn: The turn to store.
        """
        await asyncio.to_thread(self._add_turn_blocking, turn)
        log.info("memory_turn_stored", role=turn.role, content_length=len(turn.content))

    def _add_turn_blocking(self, turn: ConversationTurn) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO conversation_turns (role, content, timestamp) VALUES (?, ?, ?)",
                (turn.role, turn.content, turn.timestamp.isoformat()),
            )

    async def get_recent(self, limit: int = 20) -> list[ConversationTurn]:
        """Fetch the most recent conversation turns, oldest first.

        Args:
            limit: Maximum number of turns to return.

        Returns:
            Recent turns, ordered oldest to newest.
        """
        rows = await asyncio.to_thread(self._get_recent_blocking, limit)
        return [
            ConversationTurn(role=r[0], content=r[1], timestamp=datetime.fromisoformat(r[2]))
            for r in rows
        ]

    def _get_recent_blocking(self, limit: int) -> list[tuple[str, str, str]]:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT role, content, timestamp FROM conversation_turns ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return list(reversed(cursor.fetchall()))
