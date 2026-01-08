"""Sessions Page - Save and Load Game Sessions.

This page allows users to:
- View all saved sessions
- Load a previous session to continue playing
- Create new sessions
- Delete old sessions
- Sessions include full game state + chat history
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from dnd_manager.core.logging import get_logger
from dnd_manager.models.ecs import GameState
from dnd_manager.storage.database import SessionRecord, get_database
from dnd_manager.ui.theme import Colors, apply_theme

logger = get_logger(__name__)


# =============================================================================
# Page Configuration
# =============================================================================


st.set_page_config(
    page_title="Sessions | DungeonAI",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()


# =============================================================================
# Session State
# =============================================================================


def init_session_state() -> None:
    """Initialize session state."""
    if "session_message" not in st.session_state:
        st.session_state.session_message = None
    
    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = None


# =============================================================================
# Helper Functions
# =============================================================================


def format_date(dt: datetime) -> str:
    """Format datetime for display."""
    now = datetime.now()
    delta = now - dt
    
    if delta.days == 0:
        if delta.seconds < 3600:
            minutes = delta.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        hours = delta.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif delta.days == 1:
        return "Yesterday"
    elif delta.days < 7:
        return f"{delta.days} days ago"
    else:
        return dt.strftime("%b %d, %Y")


def get_session_summary(session: SessionRecord) -> dict[str, Any]:
    """Get a summary of a session's contents."""
    try:
        state_dict = session.get_game_state_dict()
        chat_history = session.get_chat_history()
        
        # Count actors by type
        actors = state_dict.get("actors", {})
        player_count = sum(1 for a in actors.values() if a.get("type") == "player")
        monster_count = sum(1 for a in actors.values() if a.get("type") == "monster")
        
        return {
            "player_count": player_count,
            "monster_count": monster_count,
            "message_count": len(chat_history),
            "combat_active": state_dict.get("combat_active", False),
            "combat_round": state_dict.get("combat_round", 0),
            "party_xp": state_dict.get("party_xp", 0),
        }
    except Exception:
        return {
            "player_count": 0,
            "monster_count": 0,
            "message_count": 0,
            "combat_active": False,
            "combat_round": 0,
            "party_xp": 0,
        }


# =============================================================================
# Main Page
# =============================================================================


def main() -> None:
    """Render the sessions page."""
    init_session_state()
    db = get_database()
    
    # Header
    st.title("💾 Sessions")
    st.markdown("""
    Manage your saved game sessions. Each session preserves your characters, 
    story progress, and chat history.
    """)
    
    # Show message if any
    if st.session_state.session_message:
        msg_type, msg_text = st.session_state.session_message
        if msg_type == "success":
            st.success(msg_text)
        elif msg_type == "error":
            st.error(msg_text)
        elif msg_type == "info":
            st.info(msg_text)
        st.session_state.session_message = None
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🆕 New Session", key="new_session_btn", use_container_width=True):
            st.session_state.show_new_session_dialog = True
    
    with col2:
        if st.button("🔄 Refresh", key="refresh_btn", use_container_width=True):
            st.rerun()
    
    # New session dialog
    if st.session_state.get("show_new_session_dialog", False):
        render_new_session_dialog()
    
    st.divider()
    
    # Sessions list
    render_sessions_list()


def get_available_characters() -> dict[str, Any]:
    """Load available characters from storage."""
    import json
    db = get_database()
    
    session = db.get_session("__characters__")
    if session:
        try:
            return json.loads(session.game_state_json)
        except Exception:
            pass
    return {}


def get_available_adventures() -> list[str]:
    """Get list of indexed adventure modules."""
    db = get_database()
    rulebooks = db.get_all_rulebooks()
    
    return [r.name for r in rulebooks if r.doc_type == "adventure"]


def render_new_session_dialog() -> None:
    """Render the new session creation dialog."""
    st.markdown("### Create New Session")
    st.markdown("*Select a character and campaign to begin*")
    
    # Load available data
    characters = get_available_characters()
    adventures = get_available_adventures()
    
    # Character selection (required)
    st.markdown("#### 🧙 Select Character")
    
    if not characters:
        st.warning("No characters available. Go to **Characters** to create or import one first.")
        if st.button("Go to Characters →", key="goto_chars"):
            st.switch_page("pages/2_Characters.py")
        return
    
    char_options = {uid: data.get("name", "Unknown") for uid, data in characters.items()}
    selected_char_uid = st.selectbox(
        "Character",
        options=list(char_options.keys()),
        format_func=lambda x: char_options[x],
        key="new_session_char",
    )
    
    # Show selected character preview
    if selected_char_uid:
        char_data = characters[selected_char_uid]
        col1, col2, col3 = st.columns(3)
        col1.metric("Race", char_data.get("race", "Unknown"))
        
        class_info = "Unknown"
        if char_data.get("class_features") and char_data["class_features"].get("classes"):
            classes = char_data["class_features"]["classes"]
            class_info = ", ".join(f"{c['class_name']} {c['level']}" for c in classes)
        col2.metric("Class", class_info)
        
        hp = char_data.get("health", {})
        col3.metric("HP", f"{hp.get('hp_max', '?')}")
    
    st.divider()
    
    # Campaign selection
    st.markdown("#### 📖 Select Campaign")
    
    campaign_mode = st.radio(
        "Campaign Source",
        ["Use indexed adventure", "Custom campaign name"],
        key="campaign_mode",
        horizontal=True,
    )
    
    if campaign_mode == "Use indexed adventure":
        if not adventures:
            st.info("No adventures indexed. You can add adventure modules in the **Library**, or use a custom campaign name.")
            campaign_name = st.text_input(
                "Campaign Name",
                placeholder="e.g., Waterdeep: Dragon Heist",
                key="new_campaign_name_fallback",
            )
        else:
            campaign_name = st.selectbox(
                "Adventure",
                options=adventures,
                key="new_session_adventure",
            )
    else:
        campaign_name = st.text_input(
            "Campaign Name",
            placeholder="e.g., Waterdeep: Dragon Heist",
            key="new_campaign_name_custom",
        )
    
    st.divider()
    
    # Session details
    st.markdown("#### 📝 Session Details")
    
    session_name = st.text_input(
        "Session Name",
        placeholder="e.g., Session 1 - The Beginning",
        key="new_session_name",
    )
    
    description = st.text_area(
        "Description (optional)",
        placeholder="Brief notes about this session...",
        key="new_session_desc",
        height=80,
    )
    
    st.divider()
    
    # Create / Cancel buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Create Session", key="create_session_btn", use_container_width=True):
            if not session_name.strip():
                st.error("Please enter a session name")
            elif not campaign_name or not campaign_name.strip():
                st.error("Please select or enter a campaign name")
            elif not selected_char_uid:
                st.error("Please select a character")
            else:
                create_new_session(
                    name=session_name.strip(),
                    campaign_name=campaign_name.strip(),
                    description=description.strip(),
                    character_uid=selected_char_uid,
                    character_data=characters[selected_char_uid],
                )
    
    with col2:
        if st.button("❌ Cancel", key="cancel_create_btn", use_container_width=True):
            st.session_state.show_new_session_dialog = False
            st.rerun()


def generate_campaign_opening(campaign_name: str, character_name: str) -> str:
    """Generate an opening message for the campaign using RAG content if available."""
    from pathlib import Path
    from dnd_manager.ingestion.universal_loader import ChromaStore, DocumentType
    import re
    
    def clean_and_extract_player_content(content: str) -> str:
        """Extract only player-facing content, removing DM instructions."""
        
        # DM instruction patterns to remove
        dm_patterns = [
            r"The sequence of events.*?(?=\n\n|\Z)",
            r"You can present them.*?(?=\n\n|\Z)",
            r"You can adapt.*?(?=\n\n|\Z)",
            r"The only exception is.*?(?=\n\n|\Z)",
            r"GENERAL FEATURES.*",  # Everything after this is usually DM notes
            r"Light\..*?(?=\n\n|\Z)",  # DM lighting descriptions
            r"This section describes.*?(?=\n\n|\Z)",
        ]
        
        for pattern in dm_patterns:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean OCR artifacts
        content = re.sub(r'\s*\[.*?\]\s*', ' ', content)
        content = re.sub(r'\s*[·•■□]\s*', ' ', content)
        content = re.sub(r'[^\x00-\x7F]+', ' ', content)
        
        # Fix weird line breaks (single newlines in middle of sentences)
        content = re.sub(r'(?<=[a-z,])\n(?=[a-z])', ' ', content)
        
        # Fix multiple spaces/newlines
        content = re.sub(r' {2,}', ' ', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    def find_scene_start(content: str) -> str:
        """Find where the actual scene description starts."""
        # Look for common adventure opening patterns
        scene_starters = [
            r"For the past several days",
            r"As you approach",
            r"You have been traveling",
            r"The adventure begins",
            r"You find yourself",
            r"When the adventure starts",
        ]
        
        for starter in scene_starters:
            match = re.search(starter, content, re.IGNORECASE)
            if match:
                return content[match.start():]
        
        return content
    
    # Try to find adventure content in RAG
    chroma_path = Path.home() / ".dungeonai" / "chroma"
    if chroma_path.exists():
        try:
            chroma_store = ChromaStore(persist_directory=str(chroma_path))
            
            # Search for the actual opening scene
            search_queries = [
                f"chapter 1 {campaign_name} past several days traveling",
                f"{campaign_name} greenest flames dragon attack",
                f"{campaign_name} opening scene read aloud",
            ]
            
            results = []
            seen_ids = set()
            for query in search_queries:
                docs = chroma_store.search(query, n_results=5, doc_type=DocumentType.ADVENTURE)
                for doc in docs:
                    if doc.chunk_id not in seen_ids:
                        content_lower = doc.content.lower()
                        
                        # Skip meta content
                        skip_keywords = [
                            "appendix", "background template", "stat block",
                            "monster manual", "player's handbook", 
                            "this adventure is designed", "welcome to tyranny",
                        ]
                        if any(kw in content_lower for kw in skip_keywords):
                            continue
                        
                        # Prefer content with actual scene descriptions
                        has_scene = any(phrase in content_lower for phrase in [
                            "for the past several days",
                            "you see the town",
                            "columns of black smoke",
                            "dragon",
                        ])
                        
                        seen_ids.add(doc.chunk_id)
                        results.append((doc, has_scene))
            
            # Sort to prefer content with scene descriptions
            results.sort(key=lambda x: x[1], reverse=True)
            
            if results:
                best_content = results[0][0].content
                
                # Find where the actual scene starts
                best_content = find_scene_start(best_content)
                
                # Clean and remove DM instructions
                best_content = clean_and_extract_player_content(best_content)
                
                # Find a good ending point (end of the read-aloud text)
                # Usually ends before "The sequence" or similar DM notes
                end_markers = [
                    "attacked by a dragon!",
                    "Greenest is being attacked",
                    "wheeling low over the keep",
                ]
                
                for marker in end_markers:
                    idx = best_content.lower().find(marker.lower())
                    if idx != -1:
                        end_idx = idx + len(marker)
                        # Include the punctuation
                        if end_idx < len(best_content) and best_content[end_idx] in '.!':
                            end_idx += 1
                        best_content = best_content[:end_idx]
                        break
                
                # Final length limit
                if len(best_content) > 1000:
                    break_point = best_content.rfind('.', 0, 1000)
                    if break_point > 300:
                        best_content = best_content[:break_point + 1]
                
                if best_content:
                    return f"""# 📜 {campaign_name}

---

{best_content}

---

**{character_name}**, you stand at the precipice of adventure.

*What do you do?*"""
                    
        except Exception as e:
            logger.warning(f"Failed to get adventure content: {e}")
    
    # Generic opening if no adventure content found
    return f"""# 🎭 {campaign_name}

---

Greetings, **{character_name}**! I am your Dungeon Master for **{campaign_name}**.

The world stretches before you, full of mysteries to uncover and challenges to overcome. 

*What would you like to do?*"""


def create_new_session(
    name: str,
    campaign_name: str,
    description: str,
    character_uid: str,
    character_data: dict,
) -> None:
    """Create a new game session with character and campaign."""
    from dnd_manager.models.ecs import ActorEntity
    
    db = get_database()
    
    # Create fresh game state with the selected character
    game_state = GameState(campaign_name=campaign_name)
    
    # Add the character to the game state
    character = ActorEntity.model_validate(character_data)
    game_state.add_actor(character)
    game_state.party_uids.append(character.uid)
    
    # Generate campaign opening message
    opening_message = generate_campaign_opening(campaign_name, character.name)
    chat_history = [{"role": "dm", "content": opening_message}]
    
    # Save to database
    session = db.save_session(
        name=name,
        campaign_name=campaign_name,
        game_state_dict=game_state.model_dump(mode="json"),
        chat_history=chat_history,
        description=description,
    )
    
    # Store session ID and game state for the Play page
    st.session_state.active_session_id = session.id
    st.session_state.active_character_uid = character.uid
    st.session_state.loaded_game_state = game_state.model_dump(mode="json")
    st.session_state.loaded_chat_history = chat_history
    st.session_state.show_new_session_dialog = False
    st.session_state.session_message = ("success", f"Created session **{name}** with **{character.name}**. Go to Play to start!")
    
    logger.info(f"Created new session: {name} (id={session.id}) with character {character.name}")
    st.rerun()


def render_sessions_list() -> None:
    """Render the list of saved sessions."""
    db = get_database()
    all_sessions = db.get_all_sessions()
    
    # Filter out internal storage sessions (by ID prefix or system campaign)
    sessions = [
        s for s in all_sessions 
        if not s.id.startswith("__") and s.campaign_name != "__system__"
    ]
    
    if not sessions:
        st.info("""
        No saved sessions yet. Click **New Session** to start your first adventure!
        
        **What gets saved?**
        - All characters and their stats
        - NPCs and monsters in the scene
        - Combat state and initiative order
        - Complete chat history with the DM
        - Scene descriptions and notes
        """)
        return
    
    # Summary
    st.markdown(f"**{len(sessions)}** saved session{'s' if len(sessions) != 1 else ''}")
    
    # Render each session as a card
    for session in sessions:
        render_session_card(session)


def render_session_card(session: SessionRecord) -> None:
    """Render a single session card."""
    summary = get_session_summary(session)
    
    # Container with border
    with st.container():
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.markdown(f"### {session.name}")
            st.markdown(f"📖 *{session.campaign_name}*")
            if session.description:
                st.caption(session.description[:100] + "..." if len(session.description) > 100 else session.description)
        
        with col2:
            # Session stats
            st.markdown(f"""
            👥 **{summary['player_count']}** player{'s' if summary['player_count'] != 1 else ''}
            · 💬 **{summary['message_count']}** messages
            """)
            
            if summary['combat_active']:
                st.markdown(f"⚔️ **In Combat** (Round {summary['combat_round']})")
            
            st.caption(f"Last played: {format_date(session.updated_at)}")
        
        with col3:
            # Action buttons
            if st.button("▶️ Load", key=f"load_{session.id}", use_container_width=True):
                load_session(session)
            
            # Delete with confirmation
            if st.session_state.confirm_delete == session.id:
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✓", key=f"confirm_del_{session.id}"):
                        delete_session(session)
                with col_no:
                    if st.button("✗", key=f"cancel_del_{session.id}"):
                        st.session_state.confirm_delete = None
                        st.rerun()
            else:
                if st.button("🗑️", key=f"delete_{session.id}", use_container_width=True):
                    st.session_state.confirm_delete = session.id
                    st.rerun()
        
        st.divider()


def load_session(session: SessionRecord) -> None:
    """Load a session for play."""
    try:
        # Parse the saved state
        state_dict = session.get_game_state_dict()
        chat_history = session.get_chat_history()
        
        # Store in session state for the Play page
        st.session_state.active_session_id = session.id
        st.session_state.loaded_game_state = state_dict
        st.session_state.loaded_chat_history = chat_history
        st.session_state.session_message = ("success", f"Loaded **{session.name}**. Go to Play to continue!")
        
        logger.info(f"Loaded session: {session.name} (id={session.id})")
        st.rerun()
        
    except Exception as e:
        logger.exception("Failed to load session")
        st.session_state.session_message = ("error", f"Failed to load session: {e}")
        st.rerun()


def delete_session(session: SessionRecord) -> None:
    """Delete a session."""
    db = get_database()
    
    if db.delete_session(session.id):
        st.session_state.confirm_delete = None
        st.session_state.session_message = ("success", f"Deleted **{session.name}**")
        
        # Clear active session if it was the deleted one
        if st.session_state.get("active_session_id") == session.id:
            st.session_state.active_session_id = None
        
        logger.info(f"Deleted session: {session.name} (id={session.id})")
    else:
        st.session_state.session_message = ("error", "Failed to delete session")
    
    st.rerun()


# =============================================================================
# Sidebar
# =============================================================================


def render_sidebar() -> None:
    """Render the sidebar with current session info."""
    with st.sidebar:
        st.markdown("# 💾 Sessions")
        st.markdown("Save and load your adventures.")
        
        st.divider()
        
        # Current session info
        if st.session_state.get("active_session_id"):
            db = get_database()
            session = db.get_session(st.session_state.active_session_id)
            
            if session:
                st.markdown("### 🎮 Active Session")
                st.markdown(f"**{session.name}**")
                st.caption(session.campaign_name)
                
                summary = get_session_summary(session)
                st.metric("Messages", summary['message_count'])
                st.metric("Party XP", summary['party_xp'])
        else:
            st.info("No active session. Create or load a session to play.")
        
        st.divider()
        
        # Stats
        db = get_database()
        st.markdown("### 📊 Stats")
        st.metric("Total Sessions", db.get_session_count())
        st.metric("Rulebooks", db.get_rulebook_count())
        
        st.divider()
        
        # Help
        st.markdown("### ❓ Help")
        st.markdown("""
        **Sessions save:**
        - All characters & NPCs
        - Combat state
        - Complete chat history
        - Scene descriptions
        
        **Auto-save**
        Sessions save automatically 
        when you use the Save button 
        in the Play page.
        """)


# =============================================================================
# Entry Point
# =============================================================================


render_sidebar()
main()
