"""Play Page - Main Game Interface.

This page provides:
- The main roleplay chat interface with the AI DM
- Character sheet display (sidebar)
- PDF character sheet upload
- Combat controls
- Session save functionality

Entity creation is handled entirely by the AI DM using RAG data.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import streamlit as st

from dnd_manager.core.logging import get_logger
from dnd_manager.dm.orchestrator import DMOrchestrator, DMResponse
from dnd_manager.ingestion.universal_loader import ChromaStore, UniversalIngestor
from dnd_manager.models.ecs import ActorEntity, ActorType, GameState
from dnd_manager.storage.database import get_database
from dnd_manager.ui.theme import (
    Colors,
    apply_theme,
    render_ability_scores,
    render_chat_message,
    render_hp_bar,
    render_spell_slots,
)
import os

logger = get_logger(__name__)


# =============================================================================
# Page Configuration
# =============================================================================


st.set_page_config(
    page_title="Play | DungeonAI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# Custom CSS for sleek chat action buttons
st.markdown("""
<style>
/* Make icon-only buttons smaller and more subtle */
button[kind="secondary"][data-testid="baseButton-secondary"]:has(span[data-testid="stIconMaterial"]) {
    padding: 4px 8px !important;
    min-height: 32px !important;
    background: transparent !important;
    border: none !important;
    opacity: 0.5;
    transition: opacity 0.2s, background 0.2s;
}

button[kind="secondary"][data-testid="baseButton-secondary"]:has(span[data-testid="stIconMaterial"]):hover {
    opacity: 1;
    background: rgba(255, 255, 255, 0.1) !important;
}

/* Chat message action container */
.chat-actions {
    opacity: 0;
    transition: opacity 0.2s;
}

.stChatMessage:hover .chat-actions {
    opacity: 1;
}

/* Make material icons slightly smaller */
span[data-testid="stIconMaterial"] {
    font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State
# =============================================================================


def init_settings_state() -> None:
    """Initialize settings in session state from environment variables."""
    if "settings" not in st.session_state:
        # Use environment variable if set, otherwise use default model
        # The Settings page will load the actual available models from OpenRouter API
        st.session_state.settings = {
            "api_provider": "openrouter",
            "openrouter_api_key": os.environ.get("DND_MANAGER_OPENROUTER_API_KEY", ""),
            "dm_model": os.environ.get("DND_MANAGER_DM_MODEL", "google/gemini-3-pro-preview"),
            "vision_model": os.environ.get("DND_MANAGER_VISION_MODEL", "google/gemini-2.0-flash-001"),
            "auto_save": True,
            "show_dice_details": True,
            "compact_combat": False,
        }
        logger.info("Initialized settings from env", dm_model=st.session_state.settings["dm_model"])


def init_session_state() -> None:
    """Initialize all session state variables."""
    
    # Initialize settings first
    init_settings_state()
    
    # Active session ID (from Sessions page)
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = None
    
    # Active character UID - initialize early
    if "active_character_uid" not in st.session_state:
        st.session_state.active_character_uid = None
    
    # Game state - load from session if available (always check for loaded data)
    if st.session_state.get("loaded_game_state"):
        # Load from saved/new session - this takes priority
        try:
            st.session_state.game_state = GameState.model_validate(
                st.session_state.loaded_game_state
            )
            # Also reset the DM so it gets the new game state
            st.session_state.dm = None
        except Exception as e:
            logger.exception("Failed to load game state")
            st.session_state.game_state = GameState(campaign_name="New Adventure")
        finally:
            del st.session_state.loaded_game_state
    elif "game_state" not in st.session_state:
        st.session_state.game_state = GameState(campaign_name="New Adventure")
    
    # Chat history - load from session if available (always check for loaded data)
    if st.session_state.get("loaded_chat_history") is not None:
        # Load chat history from saved/new session - this takes priority
        st.session_state.chat_history = st.session_state.loaded_chat_history
        # Reset DM so it picks up the new conversation history
        st.session_state.dm = None
        del st.session_state.loaded_chat_history
    elif "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Fallback: If we have an active session but empty chat, try loading from database
    if (st.session_state.active_session_id and 
        not st.session_state.chat_history):
        try:
            db = get_database()
            session = db.get_session(st.session_state.active_session_id)
            if session:
                loaded_history = session.get_chat_history()
                if loaded_history:
                    st.session_state.chat_history = loaded_history
                    # Reset DM so it picks up the loaded conversation history
                    st.session_state.dm = None
                    logger.info(f"Loaded {len(loaded_history)} messages from database")
        except Exception as e:
            logger.warning(f"Failed to load chat history from database: {e}")
    
    # DM Orchestrator
    if "dm" not in st.session_state:
        st.session_state.dm = None
    
    # Ingestor for character sheets
    if "ingestor" not in st.session_state:
        chroma_path = Path.home() / ".dungeonai" / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        chroma_store = ChromaStore(persist_directory=str(chroma_path))
        st.session_state.ingestor = UniversalIngestor(chroma_store=chroma_store)
    
    # Processing flag
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    
    # Full sheet view
    if "show_full_sheet" not in st.session_state:
        st.session_state.show_full_sheet = False
    
    # Level up view
    if "show_level_up" not in st.session_state:
        st.session_state.show_level_up = False


def get_or_create_dm() -> DMOrchestrator:
    """Get or create the DM orchestrator."""
    if st.session_state.dm is None:
        chroma_path = Path.home() / ".dungeonai" / "chroma"
        chroma_store = ChromaStore(persist_directory=str(chroma_path))
        
        # Use ChromaStore for HyDE retrieval (already initialized above)
        # HyDE will use the same ChromaDB backend as the main RAG system
        rag_store = chroma_store
        logger.info("Using ChromaStore for HyDE retrieval")
        
        # Convert chat history to conversation format for the DM
        conversation_history = []
        for msg in st.session_state.chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Map our roles to OpenAI roles
            if role in ("player", "user"):
                conversation_history.append({"role": "user", "content": content})
            elif role in ("dm", "assistant"):
                conversation_history.append({"role": "assistant", "content": content})
            # Skip system and roll messages - they're UI-only
        
        # Get model from settings - guaranteed to exist because init_session_state calls init_settings_state
        # This respects the user's choice from the Settings page dropdown
        dm_model = st.session_state.settings.get("dm_model", "google/gemini-3-pro-preview")
        
        # Log the model being used (visible in terminal/console)
        print(f"\n{'='*60}")
        print(f"🤖 CREATING DM ORCHESTRATOR")
        print(f"{'='*60}")
        print(f"Model: {dm_model}")
        print(f"HyDE enabled: True")
        print(f"Memory enabled: True")
        print(f"{'='*60}\n")
        
        logger.info(f"Creating DM with user-selected model: {dm_model}")
        logger.info(f"Settings state: {st.session_state.settings}")
        
        st.session_state.dm = DMOrchestrator(
            game_state=st.session_state.game_state,
            chroma_store=chroma_store,
            model=dm_model,
            conversation_history=conversation_history,
            rag_store=rag_store,  # Enable HyDE and smart filtering
            enable_hyde=True,
            enable_memory=True,  # Enable SessionMemory
        )
        
        logger.info(f"Created DM with {len(conversation_history)} history messages, model: {dm_model}")
        
        # Log actually enabled features
        features = []
        if rag_store:
            features.append("HyDE retrieval")
            features.append("smart filtering")
        features.append("session memory")
        logger.info(f"Enhanced features enabled: {', '.join(features)}")
    return st.session_state.dm


# =============================================================================
# Helper Functions
# =============================================================================


def add_chat_message(role: str, content: str, msg_type: str = "text") -> None:
    """Add a message to chat history."""
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "type": msg_type,
        "timestamp": datetime.now().isoformat(),
    })


def get_active_character() -> ActorEntity | None:
    """Get the currently active player character."""
    if st.session_state.active_character_uid is None:
        party = st.session_state.game_state.get_party()
        if party:
            st.session_state.active_character_uid = party[0].uid
            return party[0]
        return None
    
    return st.session_state.game_state.get_actor(st.session_state.active_character_uid)


def save_current_session() -> bool:
    """Save the current session to database."""
    if not st.session_state.active_session_id:
        return False
    
    db = get_database()
    session = db.get_session(st.session_state.active_session_id)
    
    if not session:
        return False
    
    try:
        db.save_session(
            name=session.name,
            campaign_name=st.session_state.game_state.campaign_name,
            game_state_dict=st.session_state.game_state.model_dump(mode="json"),
            chat_history=st.session_state.chat_history,
            description=session.description,
            session_id=session.id,
        )
        return True
    except Exception as e:
        logger.exception("Failed to save session")
        return False


# =============================================================================
# Main Page
# =============================================================================


def handle_regenerate() -> None:
    """Handle regeneration of DM response from a specific point."""
    regen_idx = st.session_state.regenerate_from_idx
    st.session_state.regenerate_from_idx = None
    
    if regen_idx is None:
        return
    
    # Get the player message to regenerate from
    if regen_idx >= len(st.session_state.chat_history):
        return
    
    player_msg = st.session_state.chat_history[regen_idx]
    if player_msg["role"] not in ("player", "user"):
        return
    
    # Remove all messages after and including the DM's response to this
    # Keep up to and including the player message we're regenerating from
    st.session_state.chat_history = st.session_state.chat_history[:regen_idx + 1]
    
    # Force DM to regenerate with fresh context
    st.session_state.dm = None  # Reset DM to get fresh conversation history
    
    # Regenerate the response
    try:
        dm = get_or_create_dm()
        
        with st.spinner("🐉 The DM reconsiders..."):
            response = dm.process_input(player_msg["content"])
        
        # Add tool results (dice rolls)
        for tool_result in response.tool_results:
            if "roll" in tool_result.tool_name.lower():
                add_chat_message("roll", tool_result.result)
        
        # Add DM narrative
        if response.narrative:
            add_chat_message("dm", response.narrative)
            
    except Exception as e:
        logger.exception("Error regenerating response")
        add_chat_message("system", f"⚠️ Error regenerating: {e}")
    
    st.rerun()


def main() -> None:
    """Main page entry point."""
    # Check if full sheet view is requested
    if st.session_state.show_full_sheet:
        render_full_character_sheet()
        return
    
    # Check if level up is requested
    if st.session_state.get("show_level_up", False):
        render_level_up_screen()
        return
    
    # Main game interface
    render_main_area()


def render_main_area() -> None:
    """Render the main chat interface."""
    
    # Handle regeneration request
    if st.session_state.get("regenerate_from_idx") is not None:
        handle_regenerate()
    
    # Header with session info and controls
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.title("🎭 Adventure")
        if st.session_state.active_session_id:
            db = get_database()
            session = db.get_session(st.session_state.active_session_id)
            if session:
                st.caption(f"📖 {session.campaign_name} — {session.name}")
    
    with col2:
        character = get_active_character()
        if character:
            if st.button("📋 Character", key="open_sheet", use_container_width=True):
                st.session_state.show_full_sheet = True
                st.rerun()
    
    with col3:
        if st.session_state.active_session_id:
            if st.button("💾 Save", key="save_btn", use_container_width=True):
                if save_current_session():
                    st.toast("Session saved!", icon="✅")
                else:
                    st.toast("Failed to save", icon="❌")
    
    # Scene description
    if st.session_state.game_state.scene_description:
        st.markdown(f"*{st.session_state.game_state.scene_description}*")
    
    # No session warning
    if not st.session_state.active_session_id and not st.session_state.chat_history:
        st.info("""
        👋 **Welcome to DungeonAI!**
        
        To get started:
        1. Go to **Sessions** to create or load a session
        2. Upload a **character sheet** in the sidebar
        3. Start your adventure!
        
        Or just start chatting below to begin a quick adventure.
        """)
    
    # Chat history
    render_chat_history()
    
    # Quick actions (in combat)
    if st.session_state.game_state.combat_active:
        render_combat_controls()
    
    # Chat input
    render_chat_input()


def render_chat_history() -> None:
    """Render the chat history with themed messages and edit controls.
    
    Includes pagination for long chat histories to improve performance
    and readability.
    """
    from dnd_manager.core.constants import MESSAGES_PER_PAGE
    
    # Initialize editing state
    if "editing_message_idx" not in st.session_state:
        st.session_state.editing_message_idx = None
    if "regenerate_from_idx" not in st.session_state:
        st.session_state.regenerate_from_idx = None
    if "chat_page_offset" not in st.session_state:
        st.session_state.chat_page_offset = 0
    
    # Get chat history
    history = st.session_state.chat_history
    total_messages = len(history)
    
    # Pagination logic
    if total_messages > MESSAGES_PER_PAGE + st.session_state.chat_page_offset:
        # Calculate how many earlier messages are hidden
        start_idx = total_messages - MESSAGES_PER_PAGE - st.session_state.chat_page_offset
        hidden_count = start_idx
        
        # Show "Load earlier messages" button
        if hidden_count > 0:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(
                    f"⬆️ Load earlier messages ({hidden_count} more)",
                    use_container_width=True,
                    key="load_earlier_messages",
                ):
                    # Load one page worth of earlier messages
                    st.session_state.chat_page_offset += min(MESSAGES_PER_PAGE, hidden_count)
                    st.rerun()
            st.markdown("---")
        
        # Slice the history to show only the current page
        display_history = history[start_idx:]
    else:
        display_history = history
        start_idx = 0
    
    # Render messages (with adjusted indices)
    for display_idx, msg in enumerate(display_history):
        idx = start_idx + display_idx  # Actual index in full history
        role = msg["role"]
        content = msg["content"]
        
        if role == "player" or role == "user":
            with st.chat_message("user", avatar="🧙"):
                # Check if we're editing this message
                if st.session_state.editing_message_idx == idx:
                    new_content = st.text_area(
                        "Edit message",
                        value=content,
                        key=f"edit_msg_{idx}",
                        height=100,
                    )
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        if st.button("Save", key=f"save_edit_{idx}", icon=":material/check:", type="primary"):
                            st.session_state.chat_history[idx]["content"] = new_content
                            st.session_state.editing_message_idx = None
                            st.rerun()
                    with col2:
                        if st.button("Cancel", key=f"cancel_edit_{idx}", icon=":material/close:"):
                            st.session_state.editing_message_idx = None
                            st.rerun()
                    with col3:
                        if st.button("Save & Regenerate", key=f"regen_edit_{idx}", icon=":material/refresh:"):
                            st.session_state.chat_history[idx]["content"] = new_content
                            st.session_state.regenerate_from_idx = idx
                            st.session_state.editing_message_idx = None
                            st.rerun()
                else:
                    st.markdown(content)
                    # Sleek action buttons
                    st.markdown(f"""
                    <div class="chat-actions" style="display: flex; justify-content: flex-end; gap: 4px; margin-top: 4px; opacity: 0.5;">
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Use small, icon-only buttons
                    cols = st.columns([8, 1, 1])
                    with cols[1]:
                        if st.button("", key=f"edit_btn_{idx}", help="Edit message", icon=":material/edit:"):
                            st.session_state.editing_message_idx = idx
                            st.rerun()
                    with cols[2]:
                        if st.button("", key=f"copy_btn_{idx}", help="Copy message", icon=":material/content_copy:"):
                            st.toast("Copied to clipboard!")
                            
        elif role == "dm" or role == "assistant":
            with st.chat_message("assistant", avatar="🐉"):
                st.markdown(content)
                # Sleek regenerate button
                cols = st.columns([8, 1, 1])
                with cols[1]:
                    if st.button("", key=f"regen_btn_{idx}", help="Regenerate response", icon=":material/refresh:"):
                        # Find the player message before this DM message
                        for prev_idx in range(idx - 1, -1, -1):
                            if st.session_state.chat_history[prev_idx]["role"] in ("player", "user"):
                                st.session_state.regenerate_from_idx = prev_idx
                                break
                        st.rerun()
                with cols[2]:
                    if st.button("", key=f"copy_dm_{idx}", help="Copy response", icon=":material/content_copy:"):
                        st.toast("Copied to clipboard!")
                        
        elif role == "system":
            render_chat_message("system", content)
        elif role == "roll":
            render_chat_message("roll", content)


def render_combat_controls() -> None:
    """Render combat status indicator."""
    game_state = st.session_state.game_state
    
    # Just show combat status - no quick actions
    if game_state.combat_active:
        current = game_state.get_current_combatant()
        current_name = current.name if current else "..."
        
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #6B1818, #8B2020); 
                    color: #F5EDE4; padding: 10px 16px; border-radius: 8px; 
                    text-align: center; margin: 1rem 0;">
            <div style="font-weight: 600;">⚔️ COMBAT — Round {game_state.combat_round}</div>
            <div style="font-size: 0.9em; margin-top: 4px;">Current Turn: {current_name}</div>
        </div>
        """, unsafe_allow_html=True)


def render_chat_input() -> None:
    """Render the chat input."""
    character = get_active_character()
    placeholder = f"What does {character.name} do?" if character else "Describe your action..."
    
    user_input = st.chat_input(placeholder, key="main_input")
    
    if user_input:
        process_player_input(user_input)
        st.rerun()


def process_player_input(user_input: str) -> None:
    """Process player input through the DM."""
    if st.session_state.is_processing:
        return
    
    st.session_state.is_processing = True
    
    try:
        add_chat_message("player", user_input)
        
        dm = get_or_create_dm()
        
        with st.spinner("🐉 The DM considers..."):
            response = dm.process_input(user_input)
        
        # Add tool results (dice rolls)
        for tool_result in response.tool_results:
            if "roll" in tool_result.tool_name.lower():
                add_chat_message("roll", tool_result.result)
        
        # Add DM narrative
        if response.narrative:
            add_chat_message("dm", response.narrative)
            
    except Exception as e:
        logger.exception("Error processing input")
        add_chat_message("system", f"⚠️ Error: {e}")
        
    finally:
        st.session_state.is_processing = False


# =============================================================================
# Level Up Screen
# =============================================================================


def render_level_up_screen() -> None:
    """Render the level up interface."""
    from dnd_manager.models.progression import (
        get_level_up_info,
        get_hit_die,
        is_asi_level,
        CLASS_FEATURES,
    )
    
    character = get_active_character()
    
    if not character or not character.class_features:
        st.warning("No character to level up")
        if st.button("← Back"):
            st.session_state.show_level_up = False
            st.rerun()
        return
    
    # Get class info
    primary_class = character.class_features.classes[0]
    current_level = primary_class.level
    new_level = current_level + 1
    
    if new_level > 20:
        st.warning("Already at maximum level!")
        if st.button("← Back"):
            st.session_state.show_level_up = False
            st.rerun()
        return
    
    # Get level up info
    level_info = get_level_up_info(primary_class.class_name, new_level, primary_class.subclass)
    
    # Header
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #C9A227, #FFD700); 
                color: #1A1A2E; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;">
        <h1 style="margin: 0;">🎉 Level Up!</h1>
        <h3 style="margin: 5px 0 0 0;">{character.name} → Level {new_level} {primary_class.class_name}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Hit Points
    st.markdown("### ❤️ Hit Points")
    hit_die = get_hit_die(primary_class.class_name)
    con_mod = character.stats.con_mod
    avg_hp = (hit_die // 2) + 1 + con_mod
    
    col1, col2 = st.columns(2)
    with col1:
        hp_method = st.radio(
            "HP Increase Method",
            ["Average (Recommended)", "Roll"],
            key="hp_method",
        )
    
    with col2:
        if hp_method == "Roll":
            if st.button("🎲 Roll Hit Die", key="roll_hp"):
                import random
                roll = random.randint(1, hit_die)
                st.session_state.hp_roll = roll
            
            hp_roll = st.session_state.get("hp_roll")
            if hp_roll:
                hp_gain = max(1, hp_roll + con_mod)
                st.metric(f"d{hit_die} + {con_mod} CON", f"+{hp_gain} HP", delta=f"Rolled: {hp_roll}")
            else:
                st.info(f"Click to roll 1d{hit_die}")
        else:
            st.metric(f"d{hit_die}/2 + 1 + {con_mod} CON", f"+{avg_hp} HP")
    
    # New Features
    if level_info["features"]:
        st.markdown("### ⚡ New Features")
        for feature in level_info["features"]:
            if feature == "ASI":
                st.info("🎯 **Ability Score Improvement** - See below")
            else:
                st.success(f"✨ **{feature}**")
    
    # ASI Section
    asi_choices = {}
    if level_info["is_asi_level"]:
        st.markdown("### 🎯 Ability Score Improvement")
        st.markdown("*Choose to increase ability scores by 2 points total (max 20) OR take a feat*")
        
        asi_type = st.radio(
            "Choose",
            ["Ability Score Increase (+2 to one, or +1 to two)", "Take a Feat"],
            key="asi_type",
        )
        
        if asi_type == "Ability Score Increase (+2 to one, or +1 to two)":
            col1, col2 = st.columns(2)
            
            abilities = ["strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"]
            
            with col1:
                st.markdown("**First +1:**")
                asi1 = st.selectbox(
                    "Ability 1",
                    abilities,
                    format_func=lambda x: f"{x.title()} ({getattr(character.stats, x)})",
                    key="asi1",
                    label_visibility="collapsed",
                )
                if asi1:
                    asi_choices[asi1] = asi_choices.get(asi1, 0) + 1
            
            with col2:
                st.markdown("**Second +1:**")
                asi2 = st.selectbox(
                    "Ability 2",
                    abilities,
                    format_func=lambda x: f"{x.title()} ({getattr(character.stats, x)})",
                    key="asi2",
                    label_visibility="collapsed",
                )
                if asi2:
                    asi_choices[asi2] = asi_choices.get(asi2, 0) + 1
            
            # Show preview
            if asi_choices:
                st.markdown("**Preview:**")
                for ability, increase in asi_choices.items():
                    current = getattr(character.stats, ability)
                    new_val = min(20, current + increase)
                    st.markdown(f"• {ability.title()}: {current} → **{new_val}**")
        else:
            st.info("🔮 Feat selection coming soon! For now, choose ability scores or discuss with your DM.")
    
    # Spell Slots (for casters)
    if level_info["spell_slots"]:
        st.markdown("### ✨ Spell Slots")
        slot_text = []
        for level, num in sorted(level_info["spell_slots"].items()):
            slot_text.append(f"Level {level}: {num}")
        st.markdown(", ".join(slot_text))
        
        if level_info["spells_known"] > 0:
            st.info(f"You can now know up to **{level_info['spells_known']} spells**")
    
    # Subclass selection
    selected_subclass = None
    if level_info["is_subclass_level"] and not primary_class.subclass:
        from dnd_manager.models.progression import get_subclass_options
        
        st.markdown("### 🏛️ Subclass Selection")
        st.markdown("*Choose your specialization path!*")
        
        subclass_options = get_subclass_options(primary_class.class_name)
        
        if subclass_options:
            # Create radio options with descriptions
            option_labels = [f"**{name}** - {desc}" for name, desc in subclass_options]
            
            selected_idx = st.radio(
                "Choose your subclass:",
                range(len(subclass_options)),
                format_func=lambda i: f"{subclass_options[i][0]}",
                key="subclass_select",
            )
            
            if selected_idx is not None:
                selected_subclass = subclass_options[selected_idx][0]
                st.info(f"📜 **{subclass_options[selected_idx][0]}**: {subclass_options[selected_idx][1]}")
        else:
            st.warning("Subclass options not found. Discuss with your DM.")
    
    # Confirm button
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Cancel", key="cancel_level", use_container_width=True):
            st.session_state.show_level_up = False
            if "hp_roll" in st.session_state:
                del st.session_state.hp_roll
            st.rerun()
    
    with col2:
        can_confirm = True
        if hp_method == "Roll" and "hp_roll" not in st.session_state:
            can_confirm = False
        # Must select subclass if it's a subclass level
        if level_info["is_subclass_level"] and not primary_class.subclass and not selected_subclass:
            can_confirm = False
        
        if st.button(
            "✅ Confirm Level Up",
            key="confirm_level",
            use_container_width=True,
            disabled=not can_confirm,
            type="primary",
        ):
            # Perform level up
            hp_roll = st.session_state.get("hp_roll") if hp_method == "Roll" else None
            
            results = character.level_up(
                hp_roll=hp_roll,
                asi_choices=asi_choices if asi_choices else None,
                subclass=selected_subclass,
            )
            
            # Clear state
            st.session_state.show_level_up = False
            if "hp_roll" in st.session_state:
                del st.session_state.hp_roll
            
            # Show success and save
            save_current_session()
            
            # Add announcement to chat
            features_str = ", ".join(results.get("features_gained", [])) or "None"
            subclass_str = f"\n• Subclass: {results['subclass']}" if results.get("subclass") else ""
            asi_str = ""
            if results.get("asi_applied"):
                asi_parts = [f"{k.title()} +{v}" for k, v in results["asi_applied"].items()]
                asi_str = f"\n• ASI: {', '.join(asi_parts)}"
            
            add_chat_message(
                "system",
                f"🎉 **{character.name}** has reached **Level {results['new_level']}**!\n"
                f"• HP: +{results['hp_gained']} (now {results['new_hp_max']})\n"
                f"• New Features: {features_str}{subclass_str}{asi_str}"
            )
            
            st.rerun()


# =============================================================================
# Full Character Sheet View
# =============================================================================


def render_full_character_sheet() -> None:
    """Render the full character sheet view."""
    character = get_active_character()
    
    if not character:
        st.warning("No character selected")
        if st.button("← Back", key="back_no_char"):
            st.session_state.show_full_sheet = False
            st.rerun()
        return
    
    # Back button
    if st.button("← Back to Adventure", key="close_sheet", use_container_width=True):
        st.session_state.show_full_sheet = False
        st.rerun()
    
    st.divider()
    
    # Character header
    st.markdown(f"# {character.name}")
    
    if character.class_features:
        classes = ", ".join(
            f"{c.class_name} {c.level}" + (f" ({c.subclass})" if c.subclass else "")
            for c in character.class_features.classes
        )
        st.markdown(f"**{character.race} | {classes}**")
    else:
        st.markdown(f"**{character.race}**")
    
    st.markdown(f"*{character.alignment} | {character.size}*")
    
    # Two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Combat stats
        st.markdown("### ⚔️ Combat")
        c1, c2, c3 = st.columns(3)
        c1.metric("AC", character.ac)
        c2.metric("HP", f"{character.health.hp_current}/{character.health.hp_max}")
        c3.metric("Speed", f"{character.movement.speed_walk if character.movement else 30} ft")
        
        # HP Bar
        render_hp_bar(character.health.hp_current, character.health.hp_max, character.health.hp_temp)
        
        # Ability scores
        st.markdown("### 📊 Ability Scores")
        render_ability_scores({
            "strength": character.stats.strength,
            "dexterity": character.stats.dexterity,
            "constitution": character.stats.constitution,
            "intelligence": character.stats.intelligence,
            "wisdom": character.stats.wisdom,
            "charisma": character.stats.charisma,
        })
        
        # Saving throws
        st.markdown("### 🎲 Saving Throws")
        stats = character.stats
        saves = []
        for ability in ["strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"]:
            mod = getattr(stats, f"{ability[:3]}_mod")
            # Handle both enum objects and strings in save_proficiencies
            save_prof_values = []
            for s in stats.save_proficiencies:
                if hasattr(s, 'value'):
                    save_prof_values.append(s.value.lower())
                else:
                    save_prof_values.append(str(s).lower())
            is_prof = ability.lower() in save_prof_values
            if is_prof:
                total = mod + stats.proficiency_bonus
                saves.append(f"**{ability[:3].upper()}**: +{total} ✓")
            else:
                saves.append(f"{ability[:3].upper()}: {mod:+d}")
        
        st.markdown(" | ".join(saves[:3]))
        st.markdown(" | ".join(saves[3:]))
    
    with col2:
        # Skills
        st.markdown("### 🎯 Skills")
        
        skill_ability_map = {
            "acrobatics": "dexterity", "animal_handling": "wisdom",
            "arcana": "intelligence", "athletics": "strength",
            "deception": "charisma", "history": "intelligence",
            "insight": "wisdom", "intimidation": "charisma",
            "investigation": "intelligence", "medicine": "wisdom",
            "nature": "intelligence", "perception": "wisdom",
            "performance": "charisma", "persuasion": "charisma",
            "religion": "intelligence", "sleight_of_hand": "dexterity",
            "stealth": "dexterity", "survival": "wisdom",
        }
        
        stats = character.stats
        for skill, ability in sorted(skill_ability_map.items()):
            mod = getattr(stats, f"{ability[:3]}_mod")
            prof_mult = stats.skill_proficiencies.get(skill, 0)
            total = mod + (stats.proficiency_bonus * prof_mult)
            
            skill_display = skill.replace("_", " ").title()
            if prof_mult == 2:
                st.markdown(f"**{skill_display}**: +{total} ★★")
            elif prof_mult == 1:
                st.markdown(f"**{skill_display}**: +{total} ★")
            else:
                st.markdown(f"{skill_display}: {total:+d}")
    
    # Spellcasting
    if character.spellbook and character.spellbook.spell_slots:
        st.divider()
        st.markdown("### ✨ Spellcasting")
        render_spell_slots(character.spellbook.spell_slots)
        
        if character.spellbook.spells_known:
            st.markdown("**Spells Known:**")
            spells_by_level: dict[int, list[str]] = {}
            for spell in character.spellbook.spells_known:
                lvl = spell.level
                if lvl not in spells_by_level:
                    spells_by_level[lvl] = []
                spells_by_level[lvl].append(spell.name)
            
            for lvl in sorted(spells_by_level.keys()):
                level_name = "Cantrips" if lvl == 0 else f"Level {lvl}"
                st.markdown(f"**{level_name}:** {', '.join(sorted(spells_by_level[lvl]))}")
    
    # Inventory
    if character.inventory and character.inventory.items:
        st.divider()
        st.markdown("### 🎒 Inventory")
        
        inv = character.inventory
        currencies = []
        if inv.currency_cp: currencies.append(f"{inv.currency_cp} CP")
        if inv.currency_sp: currencies.append(f"{inv.currency_sp} SP")
        if inv.currency_ep: currencies.append(f"{inv.currency_ep} EP")
        if inv.currency_gp: currencies.append(f"{inv.currency_gp} GP")
        if inv.currency_pp: currencies.append(f"{inv.currency_pp} PP")
        
        if currencies:
            st.markdown(f"**Currency:** {', '.join(currencies)}")
        
        st.markdown("**Items:**")
        for item in inv.items:
            qty = f" (x{item.quantity})" if item.quantity > 1 else ""
            equipped = " [E]" if item.is_equipped else ""
            st.markdown(f"• {item.name}{qty}{equipped}")


# =============================================================================
# Sidebar
# =============================================================================


def render_sidebar() -> None:
    """Render the sidebar with character sheet and controls."""
    with st.sidebar:
        st.markdown("# 🐉 DungeonAI")
        
        # Show active model
        if "settings" in st.session_state:
            active_model = st.session_state.settings.get("dm_model", "Unknown")
            # Extract just the model name for display (remove provider prefix)
            model_display = active_model.split("/")[-1] if "/" in active_model else active_model
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.05); 
                        border-left: 3px solid #4A9EFF; 
                        padding: 6px 10px; 
                        margin-bottom: 12px; 
                        border-radius: 4px; 
                        font-size: 0.85em;">
                🤖 Model: <code>{model_display}</code>
            </div>
            """, unsafe_allow_html=True)
        
        character = get_active_character()
        
        if character:
            render_sidebar_character_sheet(character)
            
            # Active effects (always visible, right after character info)
            render_active_effects_section()
        else:
            st.info("📜 Create a character in the **Characters** section to begin!")
        
        # Combat controls (enemies, initiative, etc.)
        render_sidebar_combat_controls()


def render_sidebar_character_sheet(character: ActorEntity) -> None:
    """Render the condensed character sheet in the sidebar."""
    
    st.markdown(f"## {character.name}")
    
    if character.class_features:
        class_info = character.class_features.primary_class
        level = character.class_features.total_level
        st.markdown(f"**{character.race} {class_info} {level}**")
    
    # Check for pending level up
    if character.pending_level_up:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #C9A227, #FFD700); 
                    color: #1A1A2E; padding: 8px; border-radius: 8px; text-align: center; 
                    font-weight: 600; margin: 8px 0; animation: pulse 2s infinite;">
            🎉 LEVEL UP AVAILABLE!
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("⬆️ Level Up Now", key="level_up_btn", use_container_width=True):
            st.session_state.show_level_up = True
            st.rerun()
    
    # HP Bar
    hp = character.health
    hp_pct = hp.hp_percentage
    hp_class = "healthy" if hp_pct > 50 else ("wounded" if hp_pct > 25 else "critical")
    
    st.markdown(f"""
    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; margin: 8px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px; color: var(--text-primary);">
            <span>❤️ HP</span>
            <span>{hp.hp_current}/{hp.hp_max}</span>
        </div>
        <div style="height: 20px; background: var(--bg-elevated); border-radius: 10px; overflow: hidden;">
            <div style="height: 100%; width: {hp_pct}%; border-radius: 10px;
                        background: {'#4A7C3F' if hp_class == 'healthy' else '#C9A227' if hp_class == 'wounded' else '#8B2020'};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # XP Progress Bar (only for player characters)
    char_type = character.type.value if hasattr(character.type, 'value') else str(character.type)
    if char_type == "player":
        from dnd_manager.models.progression import get_xp_progress, get_xp_for_next_level
        
        current_level = character.current_level
        next_level_xp = get_xp_for_next_level(current_level)
        
        if next_level_xp:
            progress, total = get_xp_progress(character.experience_points, current_level)
            xp_pct = (progress / max(total, 1)) * 100
            
            st.markdown(f"""
            <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; margin: 8px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px; color: var(--text-primary);">
                    <span>⭐ XP</span>
                    <span>{character.experience_points} / {next_level_xp}</span>
                </div>
                <div style="height: 12px; background: var(--bg-elevated); border-radius: 6px; overflow: hidden;">
                    <div style="height: 100%; width: {xp_pct}%; border-radius: 6px;
                                background: linear-gradient(90deg, #6B4BA3, #9B59B6);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; margin: 8px 0;">
                <div style="display: flex; justify-content: space-between; color: var(--text-primary);">
                    <span>⭐ XP</span>
                    <span>{character.experience_points} (MAX LEVEL)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # AC and Speed
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🛡️ AC", character.ac)
    with col2:
        speed = character.movement.speed_walk if character.movement else 30
        st.metric("👟 Speed", f"{speed}")
    
    # Conditions
    if character.health.conditions:
        st.markdown("**Conditions:**")
        for condition in character.health.conditions:
            cond_name = condition.value if hasattr(condition, 'value') else str(condition)
            st.warning(f"🔴 {cond_name.title()}")
    
    # Interactive Equipment Manager
    if character.inventory and character.inventory.items:
        equipped = [item for item in character.inventory.items if item.equipped]
        unequipped = [item for item in character.inventory.items if not item.equipped]
        
        with st.expander("🗡️ Equipment", expanded=False):
            # Show equipped items with unequip buttons
            if equipped:
                st.markdown("**Currently Equipped:**")
                for i, item in enumerate(equipped):
                    col1, col2 = st.columns([3, 1])
                    
                    details = []
                    if item.damage_dice:
                        details.append(f"{item.damage_dice}")
                    if item.ac_base:
                        details.append(f"AC {item.ac_base}")
                    if item.ac_bonus:
                        details.append(f"+{item.ac_bonus} AC")
                    detail_str = f" *({', '.join(details)})*" if details else ""
                    
                    col1.markdown(f"⚔️ **{item.name}**{detail_str}")
                    if col2.button("✖", key=f"unequip_{character.uid}_{i}", help="Unequip"):
                        item.equipped = False
                        save_current_session()
                        st.rerun()
            else:
                st.caption("Nothing equipped")
            
            # Show unequipped items with equip buttons
            if unequipped:
                st.divider()
                st.markdown("**Inventory:**")
                for i, item in enumerate(unequipped):
                    col1, col2 = st.columns([3, 1])
                    
                    qty_str = f" (x{item.quantity})" if item.quantity > 1 else ""
                    details = []
                    if item.damage_dice:
                        details.append(f"{item.damage_dice}")
                    if item.ac_base:
                        details.append(f"AC {item.ac_base}")
                    if item.ac_bonus:
                        details.append(f"+{item.ac_bonus} AC")
                    detail_str = f" *({', '.join(details)})*" if details else ""
                    
                    col1.markdown(f"• {item.name}{qty_str}{detail_str}")
                    if col2.button("⬆", key=f"equip_{character.uid}_{i}", help="Equip"):
                        item.equipped = True
                        save_current_session()
                        st.rerun()
            
            # Currency display
            if character.inventory.currency_cp > 0:
                st.divider()
                cp = character.inventory.currency_cp
                gp = cp // 100
                sp = (cp % 100) // 10
                copper = cp % 10
                currency_parts = []
                if gp: currency_parts.append(f"{gp} gp")
                if sp: currency_parts.append(f"{sp} sp")
                if copper: currency_parts.append(f"{copper} cp")
                st.markdown(f"💰 **Currency:** {', '.join(currency_parts) if currency_parts else '0'}")
    
    # Spellcasting (if character has spells)
    if character.spellbook:
        spell_slots = character.spellbook.spell_slots
        cantrips = character.spellbook.cantrips
        spells_known = character.spellbook.spells_known
        spells_prepared = character.spellbook.spells_prepared
        
        has_spells = spell_slots or cantrips or spells_known or spells_prepared
        
        if has_spells:
            with st.expander("✨ Spellcasting", expanded=False):
                # Spell slots as visual indicators
                if spell_slots:
                    st.markdown("**Spell Slots:**")
                    for level in sorted(spell_slots.keys()):
                        current, max_slots = spell_slots[level]
                        filled = "●" * current
                        empty = "○" * (max_slots - current)
                        st.markdown(f"Level {level}: {filled}{empty}")
                
                # Cantrips
                if cantrips:
                    st.markdown(f"**Cantrips:** {', '.join(cantrips[:5])}")
                    if len(cantrips) > 5:
                        st.markdown(f"*...and {len(cantrips) - 5} more*")
                
                # Spells (prefer prepared if available, else known)
                active_spells = spells_prepared if spells_prepared else spells_known
                if active_spells:
                    label = "Prepared" if spells_prepared else "Known"
                    st.markdown(f"**{label}:** {', '.join(active_spells[:6])}")
                    if len(active_spells) > 6:
                        st.markdown(f"*...and {len(active_spells) - 6} more*")
    
    # Class features/abilities
    if character.class_features:
        simple_features = character.class_features.features
        tracked_features = character.class_features.tracked_features
        resources = character.class_features.resources
        
        if simple_features or tracked_features or resources:
            with st.expander("⚡ Abilities", expanded=False):
                # Tracked features with uses (new system)
                if tracked_features:
                    st.markdown("**Limited-Use Abilities:**")
                    for feat in tracked_features.values():
                        if feat.uses_max:
                            filled = "●" * (feat.uses_current or 0)
                            empty = "○" * (feat.uses_max - (feat.uses_current or 0))
                            recharge_val = feat.recharge.value if hasattr(feat.recharge, 'value') else str(feat.recharge)
                            recharge = recharge_val.replace("_", " ").title()
                            color = "#4A7C3F" if (feat.uses_current or 0) > 0 else "#8B2020"
                            st.markdown(f"""
                            <div style="margin: 3px 0;">
                                <span style="color: {color};">{filled}{empty}</span> 
                                <strong>{feat.name}</strong>
                                <span style="color: var(--text-secondary); font-size: 0.8em;">({recharge})</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"• {feat.name} *(at will)*")
                
                # Legacy resources (Ki, Rage, etc.)
                if resources:
                    st.markdown("**Resources:**")
                    for name, (current, max_val) in resources.items():
                        filled = "●" * current
                        empty = "○" * (max_val - current)
                        st.markdown(f"{name.title()}: {filled}{empty}")
                
                # Simple features (at-will, passive)
                if simple_features:
                    st.markdown("**Passive/At-Will:**")
                    for feat in simple_features[:6]:
                        st.markdown(f"• {feat}")
                    if len(simple_features) > 6:
                        st.markdown(f"*...and {len(simple_features) - 6} more*")


def render_active_effects_section() -> None:
    """Render active effects (buffs/debuffs) for the player character."""
    game_state = st.session_state.game_state
    character = get_active_character()
    
    if not character:
        return
    
    # Always show the header
    st.markdown("### ✨ Active Effects")
    
    # Collect effects from all party members
    party_effects: list[tuple[str, Any]] = []  # (actor_name, effect)
    concentration_effects: list[tuple[str, Any]] = []
    
    for uid in game_state.party_uids:
        actor = game_state.actors.get(uid)
        if actor and actor.active_effects:
            for effect in actor.active_effects:
                if effect.concentration:
                    concentration_effects.append((actor.name, effect))
                else:
                    party_effects.append((actor.name, effect))
    
    if not party_effects and not concentration_effects:
        st.markdown("""
        <div style="color: var(--text-secondary); font-size: 0.9em; padding: 10px; 
                    background: var(--bg-card); border-radius: 6px; text-align: center;">
            No active effects
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        return
    
    def render_effect_card(actor_name: str, effect: Any) -> None:
        """Render a single effect card."""
        # Get duration info
        dur_type = effect.duration_type
        if hasattr(dur_type, 'value'):
            dur_type = dur_type.value
        
        # Calculate duration remaining and progress
        dur_str = ""
        progress_pct = 100
        urgent = False
        
        if dur_type == "until_start_of_next_turn":
            dur_str = "⏱️ Until next turn"
            progress_pct = 100  # Always full, expires soon
            urgent = True
        elif dur_type == "rounds":
            if game_state.combat_active and effect.expires_round:
                rounds_left = max(0, effect.expires_round - game_state.combat_round)
                total_rounds = effect.duration_value if effect.duration_value > 0 else 10
                progress_pct = (rounds_left / total_rounds) * 100
                dur_str = f"⏱️ {rounds_left} round{'s' if rounds_left != 1 else ''}"
                urgent = rounds_left <= 1
            else:
                dur_str = "⏱️ Combat duration"
        elif dur_type == "minutes":
            dur_str = f"⏱️ {effect.duration_value} minute{'s' if effect.duration_value != 1 else ''}"
            # Can't track real minutes, show as stable
            progress_pct = 80
        elif dur_type == "hours":
            dur_str = f"⏱️ {effect.duration_value} hour{'s' if effect.duration_value != 1 else ''}"
            progress_pct = 90
        elif dur_type == "until_short_rest":
            dur_str = "☕ Until short rest"
            progress_pct = 100
        elif dur_type == "until_long_rest":
            dur_str = "🌙 Until long rest"
            progress_pct = 100
        elif dur_type == "until_dispelled":
            dur_str = "♾️ Until dispelled"
            progress_pct = 100
        else:
            dur_str = str(dur_type).replace("_", " ").title()
        
        # Color based on effect type
        eff_type = effect.effect_type
        if hasattr(eff_type, 'value'):
            eff_type = eff_type.value
        
        if eff_type in ("ac_bonus", "ac_set", "damage_resistance"):
            color = "#4A7C3F"  # Green for defensive
            icon = "🛡️"
        elif eff_type == "temp_hp":
            color = "#2E7D32"  # Darker green
            icon = "💚"
        elif eff_type in ("invisibility",):
            color = "#5C4F8C"  # Purple for utility
            icon = "👻"
        elif eff_type in ("flying",):
            color = "#5C4F8C"
            icon = "🪽"
        elif eff_type in ("save_bonus", "advantage_attacks"):
            color = "#C9A227"  # Gold for buffs
            icon = "⬆️"
        else:
            color = "#C9A227"
            icon = "✨"
        
        # Urgent styling
        border_style = f"border: 2px solid {color}; animation: pulse 1s infinite;" if urgent else f"border-left: 3px solid {color};"
        
        # Concentration marker
        conc_badge = '<span style="background: #8B2020; color: white; font-size: 0.65em; padding: 1px 4px; border-radius: 3px; margin-left: 4px;">CONC</span>' if effect.concentration else ""
        
        # Progress bar for duration
        progress_color = "#8B2020" if progress_pct <= 25 else "#C9A227" if progress_pct <= 50 else color
        progress_bar = f"""
        <div style="height: 3px; background: var(--bg-elevated); border-radius: 2px; margin-top: 4px; overflow: hidden;">
            <div style="height: 100%; width: {progress_pct}%; background: {progress_color}; transition: width 0.3s;"></div>
        </div>
        """ if dur_type in ("rounds", "minutes", "hours") else ""
        
        # Value display if applicable
        value_str = ""
        if effect.value:
            if eff_type == "ac_bonus":
                value_str = f"+{effect.value} AC"
            elif eff_type == "ac_set":
                value_str = f"AC = {effect.value}"
            elif eff_type == "temp_hp":
                value_str = f"+{effect.value} temp HP"
            elif eff_type == "flying":
                value_str = f"{effect.value}ft fly"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}15, {color}08); 
                    {border_style}
                    padding: 8px 10px; margin: 4px 0; border-radius: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600;">
                    {icon} {effect.name}{conc_badge}
                </span>
                <span style="font-size: 0.75em; color: var(--text-secondary);">{dur_str}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 2px;">
                <span style="font-size: 0.75em; color: var(--text-secondary);">
                    {actor_name}{f" • {value_str}" if value_str else ""}
                </span>
            </div>
            {progress_bar}
        </div>
        """, unsafe_allow_html=True)
    
    # Render concentration effects first (most important to track)
    if concentration_effects:
        st.markdown("**🎯 Concentration:**")
        for actor_name, effect in concentration_effects:
            render_effect_card(actor_name, effect)
    
    # Render other effects
    if party_effects:
        if concentration_effects:
            st.markdown("**Other Effects:**")
        for actor_name, effect in party_effects:
            render_effect_card(actor_name, effect)
    
    st.divider()


def render_sidebar_combat_controls() -> None:
    """Render combat controls in the sidebar."""
    game_state = st.session_state.game_state
    
    # Combat visualization - show all combatants
    def get_actor_type(a):
        return a.type.value if hasattr(a.type, 'value') else str(a.type)
    enemies = [a for a in game_state.actors.values() if get_actor_type(a) != "player"]
    
    # Show initiative order when combat is active
    if game_state.combat_active and game_state.combat_order:
        st.markdown("### ⚔️ Combat")
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #6B1818, #8B2020); 
                    color: #F5EDE4; padding: 6px 10px; border-radius: 6px; text-align: center; font-weight: 600; margin-bottom: 8px;">
            ⚔️ Round {game_state.combat_round}
        </div>
        """, unsafe_allow_html=True)
        
        # Display initiative order
        st.markdown("**Initiative Order:**")
        current = game_state.get_current_combatant()
        
        for i, uid in enumerate(game_state.combat_order):
            actor = game_state.actors.get(uid)
            if actor:
                is_current = (i == game_state.combat_turn_index)
                actor_type = get_actor_type(actor)
                
                # Determine emoji and styling
                if actor.health.is_dead:
                    emoji = "💀"
                    style = "text-decoration: line-through; opacity: 0.5;"
                elif not actor.health.is_conscious:
                    emoji = "😵"
                    style = "opacity: 0.7;"
                elif actor_type == "player":
                    emoji = "🧙"
                    style = ""
                else:
                    emoji = "👹"
                    style = ""
                
                # Current turn highlight
                if is_current:
                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, #C9A227, #D4AF37); 
                                color: #1A1410; padding: 4px 8px; border-radius: 4px; 
                                font-weight: 600; margin: 2px 0; {style}">
                        ➤ {emoji} {actor.name} (Init: {actor.initiative})
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="padding: 2px 8px; margin: 1px 0; font-size: 0.9em; {style}">
                        {emoji} {actor.name} ({actor.initiative})
                    </div>
                    """, unsafe_allow_html=True)
        
        st.divider()
    
    # Enemy HP bars (outside of initiative display)
    if enemies:
        st.markdown("### 👹 Enemies")
        
        # List all enemies with HP bars and morale
        for enemy in enemies:
            hp = enemy.health
            hp_pct = (hp.hp_current / max(hp.hp_max, 1)) * 100
            
            # Status emoji based on morale and health
            if hp.is_dead:
                status = "💀"
                bar_color = "#444"
                morale_text = ""
            elif enemy.has_surrendered:
                status = "🏳️"
                bar_color = "#5C4F8C"  # Purple for surrendered
                morale_text = " (SURRENDERED)"
            elif enemy.is_fleeing:
                status = "🏃"
                bar_color = "#5C4F8C"  # Purple for fleeing
                morale_text = " (FLEEING)"
            elif not hp.is_conscious:
                status = "😵"
                bar_color = "#8B2020"
                morale_text = ""
            else:
                # Morale-based status
                morale = enemy.morale
                if morale <= 25:
                    status = "😨"  # Broken morale
                    morale_text = " 💔"
                elif morale <= 50:
                    status = "😟"  # Shaken
                    morale_text = " 😰"
                else:
                    status = "👹"
                    morale_text = ""
                
                # HP-based bar color
                if hp_pct > 50:
                    bar_color = "#8B2020"  # Red for enemies
                elif hp_pct > 25:
                    bar_color = "#C9A227"
                else:
                    bar_color = "#4A7C3F"  # Green = nearly dead (good for player!)
            
            # Build morale bar if not dead/fled/surrendered
            morale_bar = ""
            if not hp.is_dead and not enemy.has_surrendered and not enemy.is_fleeing:
                morale_color = "#4A7C3F" if enemy.morale > 50 else "#C9A227" if enemy.morale > 25 else "#8B2020"
                morale_bar = (
                    f'<div style="display: flex; align-items: center; gap: 4px; margin-top: 2px;">'
                    f'<span style="font-size: 0.65em; color: var(--text-secondary);">Morale:</span>'
                    f'<div style="flex: 1; height: 4px; background: var(--bg-elevated); border-radius: 2px; overflow: hidden;">'
                    f'<div style="height: 100%; width: {enemy.morale}%; background: {morale_color};"></div>'
                    f'</div>'
                    f'<span style="font-size: 0.65em; color: var(--text-secondary);">{enemy.morale}%</span>'
                    f'</div>'
                )
            
            enemy_card = (
                f'<div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; margin: 4px 0;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">'
                f'<span style="font-size: 0.85em;">{status} {enemy.name}{morale_text}</span>'
                f'<span style="font-size: 0.8em; color: var(--text-secondary);">AC {enemy.ac}</span>'
                f'</div>'
                f'<div style="height: 8px; background: var(--bg-elevated); border-radius: 4px; overflow: hidden;">'
                f'<div style="height: 100%; width: {hp_pct}%; border-radius: 4px; background: {bar_color};"></div>'
                f'</div>'
                f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                f'<span style="font-size: 0.7em; color: var(--text-secondary);">HP</span>'
                f'<span style="font-size: 0.75em; color: var(--text-secondary);">{hp.hp_current}/{hp.hp_max}</span>'
                f'</div>'
                f'{morale_bar}'
                f'</div>'
            )
            st.markdown(enemy_card, unsafe_allow_html=True)
        
        st.divider()
    elif game_state.combat_active:
        st.info("No enemies tracked")
        st.divider()
    
    # Session stats
    st.markdown("### 📊 Session")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Party", len(game_state.party_uids))
    with col2:
        st.metric("XP", game_state.party_xp)


# =============================================================================
# Entry Point
# =============================================================================


# Initialize session state first, before rendering anything
init_session_state()
render_sidebar()
main()
