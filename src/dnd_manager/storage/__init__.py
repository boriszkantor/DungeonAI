"""Storage module for DungeonAI persistence.

Provides SQLite-based storage for:
- Rulebook/expansion records (persistent across sessions)
- Saved game sessions (characters, story, chat history)
"""

from dnd_manager.storage.database import (
    Database,
    RulebookRecord,
    SessionRecord,
    get_database,
)

__all__ = [
    "Database",
    "RulebookRecord",
    "SessionRecord",
    "get_database",
]
