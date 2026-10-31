"""Persistent LangGraph checkpoints for the single-process web service."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


REPO_ROOT = Path(__file__).resolve().parents[3]


@asynccontextmanager
async def open_checkpointer(path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    """Keep the SQLite connection open while the compiled graph is in use.

    Relative paths are resolved from the repository root so starting uvicorn from
    another directory does not create a second, empty conversation database.
    """
    db_path = Path(path).expanduser()
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    db_path.touch(exist_ok=True, mode=0o600)

    async with aiosqlite.connect(str(db_path)) as connection:
        # Checkpoints contain user messages and tool arguments. Restrict the
        # deserializer to LangGraph's built-in safe types on every startup.
        serde = JsonPlusSerializer(allowed_msgpack_modules=None)
        saver = AsyncSqliteSaver(connection, serde=serde)
        await saver.setup()  # Fail startup if the checkpoint database is unusable.
        yield saver
