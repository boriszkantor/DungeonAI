"""Character list view and sidebar."""

from __future__ import annotations

import streamlit as st

from dnd_manager.models.ecs import ActorEntity

from .details import render_character_details
from .editor import render_character_editor
from .pdf_import import render_pdf_upload
from .state import save_characters


def render_character_list() -> None:
    """Render the list of saved characters."""
    
    # Upload section
    render_pdf_upload()
    
    st.divider()
    
    # Character list
    st.markdown("### 👥 Saved Characters")
    
    if not st.session_state.characters:
        st.info("No characters yet. Import a PDF or create a new character.")
        return
    
    for uid, character in st.session_state.characters.items():
        render_character_card(uid, character)


def render_character_card(uid: str, character: ActorEntity) -> None:
    """Render a character card."""
    
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 2, 0.5, 0.5])
        
        with col1:
            st.markdown(f"### {character.name}")
            
            if character.class_features:
                classes = ", ".join(
                    f"{c.class_name} {c.level}"
                    for c in character.class_features.classes
                )
                st.markdown(f"**{character.race} {classes}**")
            else:
                st.markdown(f"**{character.race}**")
        
        with col2:
            st.markdown(f"**HP:** {character.health.hp_current}/{character.health.hp_max}")
            st.markdown(f"**AC:** {character.ac}")
        
        with col3:
            # Initialize editing state for this character
            edit_key = f"editing_{uid}"
            if edit_key not in st.session_state:
                st.session_state[edit_key] = False
            
            if st.button("✏️", key=f"edit_char_{uid}", help="Edit character"):
                st.session_state[edit_key] = not st.session_state[edit_key]
                st.rerun()
        
        with col4:
            if st.button("🗑️", key=f"del_char_{uid}", help="Delete character"):
                del st.session_state.characters[uid]
                save_characters()
                st.session_state.char_message = ("success", f"Deleted {character.name}")
                st.rerun()
        
        # Edit mode
        if st.session_state.get(f"editing_{uid}", False):
            with st.expander("✏️ Edit Character", expanded=True):
                render_character_editor(uid, character)
        else:
            # Expandable details
            with st.expander("View Details"):
                render_character_details(character)
        
        st.divider()


def render_sidebar() -> None:
    """Render the sidebar."""
    with st.sidebar:
        st.markdown("# 🧙 Characters")
        st.markdown("Create and manage your heroes.")
        
        st.divider()
        
        st.markdown("### 📊 Stats")
        st.metric("Total Characters", len(st.session_state.get("characters", {})))
        
        st.divider()
        
        st.markdown("### ❓ Help")
        st.markdown("""
        **Import PDF**
        Upload an existing character 
        sheet and AI will extract the stats.
        
        **Create New**
        Build a character from scratch 
        using D&D 5e rules with the 
        standard array (15,14,13,12,10,8).
        
        **Sessions**
        Characters created here can be 
        selected when starting a new 
        game session.
        """)
