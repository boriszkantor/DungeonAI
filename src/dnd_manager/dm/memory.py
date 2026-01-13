"""Memory management for DM context windows.

This module implements a summarization protocol to prevent "context rot"
in long D&D sessions. Instead of infinitely growing chat history, it:

1. Short-Term Memory: Last 10-20 raw conversation turns
2. Long-Term Memory: Summarized "Quest Log" entries stored in vector DB
3. Retrieval: When player asks about past events, retrieve from quest log

This keeps context windows manageable while maintaining campaign continuity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from dnd_manager.core.logging import get_logger


if TYPE_CHECKING:
    from dnd_manager.ingestion.rag_store import RAGStore, SearchResult

logger = get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class QuestLogEntry(BaseModel):
    """A summarized scene/encounter for long-term memory.
    
    When a scene ends, the AI summarizes it into a structured entry
    that can be stored in the vector database and retrieved semantically.
    
    Attributes:
        entry_id: Unique identifier.
        scene_name: Name/title of the scene.
        summary: Concise summary of what happened.
        npcs_met: NPCs encountered.
        locations: Locations visited.
        items_gained: Items acquired.
        quests_updated: Quest progress changes.
        key_events: Important story moments.
        timestamp: When this scene occurred.
        embedding: Vector embedding for semantic retrieval.
    """
    
    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    scene_name: str = Field(description="Scene or encounter name")
    summary: str = Field(description="What happened in this scene")
    npcs_met: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    items_gained: list[str] = Field(default_factory=list)
    quests_updated: list[str] = Field(default_factory=list)
    key_events: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    embedding: list[float] = Field(default_factory=list)
    
    def to_text(self) -> str:
        """Convert to searchable text for embedding."""
        parts = [
            f"Scene: {self.scene_name}",
            f"Summary: {self.summary}",
        ]
        
        if self.npcs_met:
            parts.append(f"NPCs: {', '.join(self.npcs_met)}")
        if self.locations:
            parts.append(f"Locations: {', '.join(self.locations)}")
        if self.items_gained:
            parts.append(f"Items: {', '.join(self.items_gained)}")
        if self.quests_updated:
            parts.append(f"Quests: {', '.join(self.quests_updated)}")
        if self.key_events:
            parts.append(f"Events: {'; '.join(self.key_events)}")
        
        return "\n".join(parts)
    
    def to_context_string(self) -> str:
        """Format for inclusion in context window."""
        return f"**{self.scene_name}**: {self.summary}"


# =============================================================================
# Session Memory Manager
# =============================================================================


class SessionMemory:
    """Manages short-term and long-term memory for D&D campaigns.
    
    This prevents "context rot" by implementing a two-tier memory system:
    
    1. **Short-Term Memory** (10-20 turns):
       - Raw conversation messages
       - Immediate context for current scene
       
    2. **Long-Term Memory** (Quest Log):
       - Summarized past scenes
       - Stored in vector DB for semantic retrieval
       - Retrieved when player asks about past events
    
    Usage:
        >>> memory = SessionMemory(rag_store, max_short_term=20)
        >>> 
        >>> # Add messages as they happen
        >>> memory.add_message({"role": "user", "content": "I explore the tavern"})
        >>> memory.add_message({"role": "assistant", "content": "You see..."})
        >>> 
        >>> # When scene ends, summarize and archive
        >>> await memory.end_scene("Tavern Encounter")
        >>> 
        >>> # Get context for next turn (includes relevant history)
        >>> context = await memory.get_context_window("Who was that bartender?")
    """
    
    def __init__(
        self,
        rag_store: RAGStore | None = None,
        *,
        max_short_term: int = 20,
        model: str = "google/gemini-2.0-flash-001",
        use_openrouter: bool = True,
    ) -> None:
        """Initialize the session memory.
        
        Args:
            rag_store: RAG store for long-term memory (quest log).
            max_short_term: Maximum messages in short-term memory.
            model: LLM model for summarization.
            use_openrouter: Whether to use OpenRouter API.
        """
        self.rag_store = rag_store
        self.max_short_term = max_short_term
        self.model = model
        self.use_openrouter = use_openrouter
        
        # Short-term memory: recent raw messages
        self.short_term: list[dict[str, Any]] = []
        
        # Long-term memory: summarized quest log entries
        self.quest_log: list[QuestLogEntry] = []
        
        # Current scene messages (to be summarized on scene end)
        self.current_scene_messages: list[dict[str, Any]] = []
        
        self._client: Any = None
        
        logger.info(
            "SessionMemory initialized",
            max_short_term=max_short_term,
            model=model,
        )
    
    def _get_client(self) -> Any:
        """Get or create the OpenAI client for summarization."""
        if self._client is None:
            try:
                from openai import OpenAI
                
                from dnd_manager.core.config import get_settings
                
                settings = get_settings()
                
                if self.use_openrouter:
                    api_key = settings.ai.openai_api_key
                    if api_key:
                        api_key = api_key.get_secret_value()
                    self._client = OpenAI(
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1",
                        default_headers={
                            "HTTP-Referer": "https://github.com/dnd-campaign-manager",
                            "X-Title": "D&D Campaign Manager",
                        },
                    )
                else:
                    api_key = settings.ai.openai_api_key
                    if api_key:
                        api_key = api_key.get_secret_value()
                    self._client = OpenAI(api_key=api_key)
                    
            except ImportError as exc:
                logger.error("openai package not installed")
                raise
        
        return self._client
    
    def add_message(self, message: dict[str, Any]) -> None:
        """Add a message to short-term memory.
        
        Args:
            message: Conversation message with "role" and "content".
        """
        self.short_term.append(message)
        self.current_scene_messages.append(message)
        
        # Trim short-term memory if too long
        if len(self.short_term) > self.max_short_term:
            # Keep most recent messages
            self.short_term = self.short_term[-self.max_short_term:]
    
    async def summarize_scene(
        self,
        scene_name: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> QuestLogEntry:
        """Summarize a scene into a quest log entry.
        
        Args:
            scene_name: Name of the scene to summarize.
            messages: Messages to summarize (defaults to current_scene_messages).
            
        Returns:
            QuestLogEntry with structured summary.
        """
        if messages is None:
            messages = self.current_scene_messages
        
        if not messages:
            # No messages to summarize
            return QuestLogEntry(
                scene_name=scene_name,
                summary="No events in this scene.",
            )
        
        # Build conversation text
        conversation_text = "\n\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages
        ])
        
        # Prompt for summarization
        system_prompt = """You are summarizing a D&D game session scene.
Extract key information into structured format:
- What happened (concise summary)
- NPCs met (names only)
- Locations visited
- Items gained
- Quest progress
- Key story events

Keep it concise but informative. Focus on facts, not prose."""

        user_prompt = f"""Summarize this D&D scene into structured data.

Scene Name: {scene_name}

Conversation:
{conversation_text[:3000]}  # Limit to prevent token overflow

Return a JSON object with:
{{
  "summary": "What happened in 2-3 sentences",
  "npcs_met": ["NPC name 1", "NPC name 2"],
  "locations": ["Location 1"],
  "items_gained": ["Item 1"],
  "quests_updated": ["Quest update 1"],
  "key_events": ["Event 1", "Event 2"]
}}"""

        try:
            client = self._get_client()
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            
            # Parse JSON response
            import json
            data = json.loads(response.choices[0].message.content or "{}")
            
            entry = QuestLogEntry(
                scene_name=scene_name,
                summary=data.get("summary", "Scene summary unavailable."),
                npcs_met=data.get("npcs_met", []),
                locations=data.get("locations", []),
                items_gained=data.get("items_gained", []),
                quests_updated=data.get("quests_updated", []),
                key_events=data.get("key_events", []),
            )
            
            logger.info(
                "Scene summarized",
                scene=scene_name,
                npcs=len(entry.npcs_met),
                locations=len(entry.locations),
            )
            
            return entry
            
        except Exception as exc:
            logger.exception("Failed to summarize scene")
            # Return basic entry on failure
            return QuestLogEntry(
                scene_name=scene_name,
                summary="Scene summary failed - see chat history.",
            )
    
    async def end_scene(self, scene_name: str) -> QuestLogEntry:
        """End the current scene and archive it to long-term memory.
        
        Args:
            scene_name: Name of the scene that's ending.
            
        Returns:
            QuestLogEntry that was created and archived.
        """
        # #region agent log H4
        import json
        with open(r'd:\Projects\D&D Campaign Manager\.cursor\debug.log', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"location":"memory.py:335","message":"end_scene called","data":{"scene_name":scene_name,"num_messages":len(self.current_scene_messages),"has_rag_store":self.rag_store is not None},"timestamp":__import__('time').time()*1000,"sessionId":"debug-session","runId":"initial","hypothesisId":"H4"})+'\n')
        # #endregion
        
        # Summarize the scene
        entry = await self.summarize_scene(scene_name, self.current_scene_messages)
        
        # Add to quest log
        self.quest_log.append(entry)
        
        # If RAG store available, index the entry for retrieval
        if self.rag_store:
            try:
                # #region agent log H4
                import json
                with open(r'd:\Projects\D&D Campaign Manager\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"location":"memory.py:357","message":"Before embedding generation","data":{"entry_text_length":len(entry.to_text()),"has_embedding_provider":hasattr(self.rag_store, 'embedding_provider')},"timestamp":__import__('time').time()*1000,"sessionId":"debug-session","runId":"initial","hypothesisId":"H4"})+'\n')
                # #endregion
                
                from dnd_manager.ingestion.rag_store import IndexedContent, ContentType
                
                # Create indexed content from quest log entry
                content = IndexedContent(
                    content_id=entry.entry_id,
                    content_type=ContentType.CUSTOM,
                    text=entry.to_text(),
                    source="quest_log",
                    title=entry.scene_name,
                    metadata={
                        "entry_type": "quest_log",
                        "npcs": entry.npcs_met,
                        "locations": entry.locations,
                        "timestamp": entry.timestamp.isoformat(),
                    },
                )
                
                # Generate embedding and add to store
                embedding = self.rag_store.embedding_provider.embed_text(content.text)
                content.embedding = embedding
                
                # #region agent log H4
                with open(r'd:\Projects\D&D Campaign Manager\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"location":"memory.py:385","message":"After embedding generation","data":{"embedding_length":len(embedding),"content_id":content.content_id},"timestamp":__import__('time').time()*1000,"sessionId":"debug-session","runId":"initial","hypothesisId":"H4"})+'\n')
                # #endregion
                
                self.rag_store.vector_store.add([content])
                
                logger.info(f"Quest log entry indexed: {scene_name}")
                
            except Exception as exc:
                # #region agent log H4
                import json
                with open(r'd:\Projects\D&D Campaign Manager\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"location":"memory.py:397","message":"Failed to index quest log","data":{"error":str(exc),"error_type":type(exc).__name__},"timestamp":__import__('time').time()*1000,"sessionId":"debug-session","runId":"initial","hypothesisId":"H4"})+'\n')
                # #endregion
                logger.warning(f"Failed to index quest log entry: {exc}")
        
        # Clear current scene messages
        self.current_scene_messages = []
        
        return entry
    
    async def get_relevant_history(
        self,
        query: str,
        *,
        k: int = 3,
    ) -> list[QuestLogEntry]:
        """Retrieve relevant past events from quest log.
        
        Args:
            query: User query about past events.
            k: Number of relevant entries to retrieve.
            
        Returns:
            List of relevant QuestLogEntry objects.
        """
        if not self.rag_store:
            # No RAG store, return recent quest log
            return self.quest_log[-k:]
        
        try:
            # Search quest log entries
            results = self.rag_store.retrieve_relevant_context(
                query,
                k=k,
                filters={"entry_type": "quest_log"},
            )
            
            # Convert back to QuestLogEntry objects
            entries = []
            for result in results:
                # Try to find in local quest log
                for entry in self.quest_log:
                    if entry.entry_id == result.content.content_id:
                        entries.append(entry)
                        break
            
            return entries
            
        except Exception as exc:
            logger.warning(f"Failed to retrieve history: {exc}")
            return self.quest_log[-k:]
    
    async def get_context_window(
        self,
        query: str,
        *,
        include_relevant_history: bool = True,
    ) -> list[dict[str, Any]]:
        """Get context window for next turn.
        
        This combines:
        1. Short-term memory (recent messages)
        2. Relevant long-term memory (if query references past events)
        
        Args:
            query: Current user query.
            include_relevant_history: Whether to retrieve from quest log.
            
        Returns:
            List of messages to include in context.
        """
        context_messages = []
        
        # Check if query references past events
        past_keywords = [
            "who was", "what happened", "where did", "when did",
            "remember", "earlier", "before", "last time",
            "that guy", "that place", "the barkeep", "the innkeeper",
        ]
        
        query_lower = query.lower()
        references_past = any(keyword in query_lower for keyword in past_keywords)
        
        # If query references past, include relevant history
        if include_relevant_history and references_past and self.quest_log:
            try:
                relevant_entries = await self.get_relevant_history(query, k=2)
                
                if relevant_entries:
                    # Add as system message
                    history_text = "**Relevant Past Events:**\n\n" + "\n\n".join([
                        entry.to_context_string() for entry in relevant_entries
                    ])
                    
                    context_messages.append({
                        "role": "system",
                        "content": history_text,
                    })
                    
                    logger.debug(
                        "Added relevant history to context",
                        entries=len(relevant_entries),
                    )
            except Exception as exc:
                logger.warning(f"Failed to add relevant history: {exc}")
        
        # Add short-term memory
        context_messages.extend(self.short_term)
        
        return context_messages


__all__ = [
    "QuestLogEntry",
    "SessionMemory",
]
