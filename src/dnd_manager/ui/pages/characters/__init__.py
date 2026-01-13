"""Characters Page - Manage Player Characters.

This page allows users to:
- Upload character sheet PDFs (vision extraction)
- Create new characters following D&D 5e rules
- View and manage saved characters
- Characters persist and can be used across sessions
"""

from __future__ import annotations

import streamlit as st

from dnd_manager.ui.theme import apply_theme

from .creator import render_character_creator
from .list_view import render_character_list, render_sidebar
from .state import init_session_state


# =============================================================================
# Main Page
# =============================================================================


def main() -> None:
    """Render the characters page."""
    init_session_state()
    
    st.title("🧙 Characters")
    st.markdown("Create and manage your player characters.")
    
    # Show message if any
    if st.session_state.char_message:
        msg_type, msg_text = st.session_state.char_message
        if msg_type == "success":
            st.success(msg_text)
        elif msg_type == "error":
            st.error(msg_text)
        st.session_state.char_message = None
    
    # Tabs for different actions
    tab1, tab2 = st.tabs(["📋 My Characters", "➕ Create New"])
    
    with tab1:
        render_character_list()
    
    with tab2:
        render_character_creator()


def setup_and_run() -> None:
    """Setup page configuration and run the main function.
    
    This should only be called from the Streamlit page file (2_Characters.py).
    """
    st.set_page_config(
        page_title="Characters | DungeonAI",
        page_icon="🧙",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    apply_theme()
    init_session_state()
    render_sidebar()
    main()


# Export the main components
__all__ = [
    "main",
    "setup_and_run",
    "render_character_creator",
    "render_character_list",
    "render_sidebar",
    "init_session_state",
]
