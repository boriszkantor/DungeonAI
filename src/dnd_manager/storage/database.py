"""SQLite persistence layer for DungeonAI.

Provides persistent storage for:
- Rulebook/expansion metadata (what has been indexed)
- Saved game sessions (full game state + chat history)

Storage location: ~/.dungeonai/dungeonai.db
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator
from uuid import UUID, uuid4

from dnd_manager.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RulebookRecord:
    """Record of an indexed rulebook/expansion.
    
    Attributes:
        id: Unique identifier.
        name: Display name of the rulebook.
        filename: Original filename.
        doc_type: Type of document (rulebook, monster_manual, etc.).
        indexed_at: When the document was indexed.
        chunk_count: Number of chunks in the vector store.
        file_hash: SHA256 hash for deduplication.
    """
    
    id: str
    name: str
    filename: str
    doc_type: str
    indexed_at: datetime
    chunk_count: int
    file_hash: str
    
    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> RulebookRecord:
        """Create from database row."""
        return cls(
            id=row[0],
            name=row[1],
            filename=row[2],
            doc_type=row[3],
            indexed_at=datetime.fromisoformat(row[4]),
            chunk_count=row[5],
            file_hash=row[6],
        )


@dataclass
class SessionRecord:
    """Record of a saved game session.
    
    Attributes:
        id: Unique session identifier.
        name: User-provided session name.
        campaign_name: Name of the campaign.
        description: Optional description.
        game_state_json: Serialized GameState.
        chat_history_json: Serialized chat history.
        created_at: When the session was created.
        updated_at: When the session was last saved.
        thumbnail: Optional base64 encoded thumbnail.
    """
    
    id: str
    name: str
    campaign_name: str
    description: str
    game_state_json: str
    chat_history_json: str
    created_at: datetime
    updated_at: datetime
    thumbnail: str | None = None
    
    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> SessionRecord:
        """Create from database row."""
        return cls(
            id=row[0],
            name=row[1],
            campaign_name=row[2],
            description=row[3],
            game_state_json=row[4],
            chat_history_json=row[5],
            created_at=datetime.fromisoformat(row[6]),
            updated_at=datetime.fromisoformat(row[7]),
            thumbnail=row[8] if len(row) > 8 else None,
        )
    
    def get_game_state_dict(self) -> dict[str, Any]:
        """Parse game state JSON."""
        return json.loads(self.game_state_json)
    
    def get_chat_history(self) -> list[dict[str, Any]]:
        """Parse chat history JSON."""
        return json.loads(self.chat_history_json)


# =============================================================================
# Database Class
# =============================================================================


class Database:
    """SQLite database for DungeonAI persistence.
    
    Manages storage of:
    - Rulebook records (what's been indexed for RAG)
    - Saved sessions (game state + chat history)
    
    Database location: ~/.dungeonai/dungeonai.db
    """
    
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize database.
        
        Args:
            db_path: Path to database file. If None, uses default location.
        """
        if db_path is None:
            self.db_path = self._get_default_path()
        else:
            self.db_path = Path(db_path)
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize schema
        self._init_schema()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    @staticmethod
    def _get_default_path() -> Path:
        """Get default database path."""
        return Path.home() / ".dungeonai" / "dungeonai.db"
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Schema version table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            
            # Rulebooks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rulebooks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE
                )
            """)
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    campaign_name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    game_state_json TEXT NOT NULL,
                    chat_history_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    thumbnail TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rulebooks_doc_type 
                ON rulebooks(doc_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_updated 
                ON sessions(updated_at DESC)
            """)
            
            # Set schema version
            cursor.execute("""
                INSERT OR REPLACE INTO schema_version (version) VALUES (?)
            """, (self.SCHEMA_VERSION,))
    
    # =========================================================================
    # Rulebook Operations
    # =========================================================================
    
    def add_rulebook(
        self,
        name: str,
        filename: str,
        doc_type: str,
        chunk_count: int,
        file_hash: str,
    ) -> RulebookRecord:
        """Add a new rulebook record.
        
        Args:
            name: Display name.
            filename: Original filename.
            doc_type: Document type.
            chunk_count: Number of indexed chunks.
            file_hash: SHA256 hash of file.
            
        Returns:
            Created rulebook record.
        """
        record_id = str(uuid4())
        indexed_at = datetime.now()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO rulebooks (id, name, filename, doc_type, indexed_at, chunk_count, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (record_id, name, filename, doc_type, indexed_at.isoformat(), chunk_count, file_hash))
        
        logger.info(f"Added rulebook: {name} ({chunk_count} chunks)")
        
        return RulebookRecord(
            id=record_id,
            name=name,
            filename=filename,
            doc_type=doc_type,
            indexed_at=indexed_at,
            chunk_count=chunk_count,
            file_hash=file_hash,
        )
    
    def get_rulebook_by_hash(self, file_hash: str) -> RulebookRecord | None:
        """Get rulebook by file hash (for deduplication).
        
        Args:
            file_hash: SHA256 hash of file.
            
        Returns:
            Rulebook record if found, None otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, filename, doc_type, indexed_at, chunk_count, file_hash
                FROM rulebooks WHERE file_hash = ?
            """, (file_hash,))
            row = cursor.fetchone()
            
            if row:
                return RulebookRecord.from_row(tuple(row))
            return None
    
    def get_all_rulebooks(self) -> list[RulebookRecord]:
        """Get all indexed rulebooks.
        
        Returns:
            List of all rulebook records.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, filename, doc_type, indexed_at, chunk_count, file_hash
                FROM rulebooks ORDER BY indexed_at DESC
            """)
            
            return [RulebookRecord.from_row(tuple(row)) for row in cursor.fetchall()]
    
    def delete_rulebook(self, rulebook_id: str) -> bool:
        """Delete a rulebook record.
        
        Note: This only removes the database record. 
        ChromaDB cleanup must be done separately.
        
        Args:
            rulebook_id: ID of rulebook to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM rulebooks WHERE id = ?", (rulebook_id,))
            deleted = cursor.rowcount > 0
        
        if deleted:
            logger.info(f"Deleted rulebook: {rulebook_id}")
        
        return deleted
    
    # =========================================================================
    # Session Operations
    # =========================================================================
    
    def save_session(
        self,
        name: str,
        campaign_name: str,
        game_state_dict: dict[str, Any],
        chat_history: list[dict[str, Any]],
        description: str = "",
        session_id: str | None = None,
        thumbnail: str | None = None,
    ) -> SessionRecord:
        """Save a game session.
        
        Args:
            name: Session name.
            campaign_name: Campaign name.
            game_state_dict: Serialized game state.
            chat_history: Chat history list.
            description: Optional description.
            session_id: Existing session ID for updates.
            thumbnail: Optional base64 thumbnail.
            
        Returns:
            Saved session record.
        """
        now = datetime.now()
        game_state_json = json.dumps(game_state_dict, default=str)
        chat_history_json = json.dumps(chat_history, default=str)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if session_id:
                # Update existing session
                cursor.execute("""
                    UPDATE sessions 
                    SET name = ?, campaign_name = ?, description = ?,
                        game_state_json = ?, chat_history_json = ?,
                        updated_at = ?, thumbnail = ?
                    WHERE id = ?
                """, (name, campaign_name, description, game_state_json, 
                      chat_history_json, now.isoformat(), thumbnail, session_id))
                
                if cursor.rowcount == 0:
                    # Session doesn't exist, create new with the provided ID
                    cursor.execute("""
                        INSERT INTO sessions 
                        (id, name, campaign_name, description, game_state_json, 
                         chat_history_json, created_at, updated_at, thumbnail)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (session_id, name, campaign_name, description, game_state_json,
                          chat_history_json, now.isoformat(), now.isoformat(), thumbnail))
                    created_at = now
                else:
                    # Get original created_at
                    cursor.execute("SELECT created_at FROM sessions WHERE id = ?", (session_id,))
                    row = cursor.fetchone()
                    created_at = datetime.fromisoformat(row[0]) if row else now
            else:
                # Create new session
                session_id = str(uuid4())
                cursor.execute("""
                    INSERT INTO sessions 
                    (id, name, campaign_name, description, game_state_json, 
                     chat_history_json, created_at, updated_at, thumbnail)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, name, campaign_name, description, game_state_json,
                      chat_history_json, now.isoformat(), now.isoformat(), thumbnail))
                created_at = now
        
        logger.info(f"Saved session: {name} (id={session_id})")
        
        return SessionRecord(
            id=session_id,
            name=name,
            campaign_name=campaign_name,
            description=description,
            game_state_json=game_state_json,
            chat_history_json=chat_history_json,
            created_at=created_at,
            updated_at=now,
            thumbnail=thumbnail,
        )
    
    def get_session(self, session_id: str) -> SessionRecord | None:
        """Get a session by ID.
        
        Args:
            session_id: Session ID.
            
        Returns:
            Session record if found, None otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, campaign_name, description, game_state_json,
                       chat_history_json, created_at, updated_at, thumbnail
                FROM sessions WHERE id = ?
            """, (session_id,))
            row = cursor.fetchone()
            
            if row:
                return SessionRecord.from_row(tuple(row))
            return None
    
    def get_all_sessions(self) -> list[SessionRecord]:
        """Get all saved sessions, sorted by last updated.
        
        Returns:
            List of all session records.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, campaign_name, description, game_state_json,
                       chat_history_json, created_at, updated_at, thumbnail
                FROM sessions ORDER BY updated_at DESC
            """)
            
            return [SessionRecord.from_row(tuple(row)) for row in cursor.fetchall()]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: ID of session to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            deleted = cursor.rowcount > 0
        
        if deleted:
            logger.info(f"Deleted session: {session_id}")
        
        return deleted
    
    def get_session_count(self, include_internal: bool = False) -> int:
        """Get total number of saved sessions.
        
        Args:
            include_internal: Include internal sessions (e.g., character storage).
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if include_internal:
                cursor.execute("SELECT COUNT(*) FROM sessions")
            else:
                cursor.execute("SELECT COUNT(*) FROM sessions WHERE id NOT LIKE '__%'")
            return cursor.fetchone()[0]
    
    def get_rulebook_count(self) -> int:
        """Get total number of indexed rulebooks."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rulebooks")
            return cursor.fetchone()[0]


# =============================================================================
# Singleton Instance
# =============================================================================


_database_instance: Database | None = None


def get_database() -> Database:
    """Get the global database instance.
    
    Returns:
        Database singleton instance.
    """
    global _database_instance
    
    if _database_instance is None:
        _database_instance = Database()
    
    return _database_instance
