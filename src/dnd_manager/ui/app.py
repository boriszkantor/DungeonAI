"""DungeonAI - Main Application Entry Point.

This is the landing page for the multi-page Streamlit application.
It provides an overview and quick navigation to the main sections:
- Library: Manage rulebooks and expansions
- Sessions: Save and load game sessions  
- Play: The main game interface
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dnd_manager.storage.database import get_database
from dnd_manager.ui.theme import Colors, apply_theme


# =============================================================================
# Page Configuration
# =============================================================================


st.set_page_config(
    page_title="DungeonAI",
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "DungeonAI - Neuro-Symbolic D&D 5E Engine",
    },
)

apply_theme()


# =============================================================================
# Main Page
# =============================================================================


def main() -> None:
    """Render the main landing page."""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">🐉 DungeonAI</h1>
        <p style="font-size: 1.25rem; color: var(--phb-text); font-style: italic;">
            Neuro-Symbolic D&D 5E Engine
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Quick stats
    db = get_database()
    rulebook_count = db.get_rulebook_count()
    session_count = db.get_session_count()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📚 Rulebooks", rulebook_count)
    with col2:
        st.metric("💾 Sessions", session_count)
    with col3:
        # Check for active session
        active = "Active" if st.session_state.get("active_session_id") else "None"
        st.metric("🎮 Current Session", active)
    
    st.divider()
    
    # Navigation cards
    st.markdown("## Get Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📚 Library
        
        Manage your D&D rulebooks, monster manuals, and adventure modules.
        Documents indexed here provide the AI DM with accurate rules and monster stats.
        
        **Use this to:**
        - Add Player's Handbook, DMG, Monster Manual
        - Index adventure modules
        - Manage your sourcebook collection
        """)
        if st.button("Open Library →", key="nav_library", use_container_width=True):
            st.switch_page("pages/1_Library.py")
    
    with col2:
        st.markdown("""
        ### 🧙 Characters
        
        Create and manage your player characters. Import existing character
        sheet PDFs or create new characters following D&D 5e rules.
        
        **Use this to:**
        - Import character sheet PDFs
        - Create new characters
        - Manage your character roster
        """)
        if st.button("Open Characters →", key="nav_characters", use_container_width=True):
            st.switch_page("pages/2_Characters.py")
    
    with col3:
        st.markdown("""
        ### 💾 Sessions
        
        Start new adventures or continue saved ones. Each session links
        a character to a campaign and preserves all progress.
        
        **Use this to:**
        - Create new game sessions
        - Load previous sessions
        - Manage saved games
        """)
        if st.button("Open Sessions →", key="nav_sessions", use_container_width=True):
            st.switch_page("pages/3_Sessions.py")
    
    # Play button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("### 🎭 Play")
        st.markdown("Continue your current adventure or start playing after creating a session.")
        if st.button("Start Playing →", key="nav_play", use_container_width=True):
            st.switch_page("pages/4_Play.py")
    
    st.divider()
    
    # Quick start guide
    st.markdown("## Quick Start Guide")
    
    st.markdown("""
    <div style="background: var(--bg-card); border: 1px solid var(--border); 
                border-radius: 8px; padding: 1.5rem; margin: 1rem 0;">
        <h4 style="color: var(--crimson); margin-top: 0;">First Time Setup</h4>
        <ol style="margin-bottom: 0; color: var(--text-primary);">
            <li><strong>Add Rulebooks:</strong> Go to <em>Library</em> and upload your D&D PDFs 
                (Player's Handbook, Monster Manual, etc.). This allows the DM to look up accurate rules and stats.</li>
            <li><strong>Create a Character:</strong> Go to <em>Characters</em> to import a character sheet PDF
                or create a new character using D&D 5e rules.</li>
            <li><strong>Start a Session:</strong> Go to <em>Sessions</em> and create a new game session.
                Select your character and campaign to begin.</li>
            <li><strong>Play!</strong> Type your actions in the chat and let the AI DM guide your adventure.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    st.markdown("## Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎲 True Dice Rolls**
        
        All dice rolls are handled by Python using the d20 library.
        The AI DM cannot fudge dice - what you roll is what you get.
        
        **📖 RAG-Powered Rules**
        
        The DM references your indexed rulebooks for accurate rulings
        on spells, abilities, and monster stats.
        """)
    
    with col2:
        st.markdown("""
        **📋 Vision Character Import**
        
        Upload a character sheet PDF and AI vision extracts all your
        stats, spells, and inventory automatically.
        
        **💾 Session Persistence**
        
        Your adventures are saved locally. Pick up exactly where you
        left off - characters, story, and chat history included.
        """)


# =============================================================================
# Sidebar
# =============================================================================


def render_sidebar() -> None:
    """Render the sidebar navigation."""
    with st.sidebar:
        st.markdown("# 🐉 DungeonAI")
        st.markdown("*AI-Powered D&D 5E*")
        
        st.divider()
        
        st.markdown("### Navigation")
        
        if st.button("📚 Library", key="side_library", use_container_width=True):
            st.switch_page("pages/1_Library.py")
        
        if st.button("🧙 Characters", key="side_characters", use_container_width=True):
            st.switch_page("pages/2_Characters.py")
        
        if st.button("💾 Sessions", key="side_sessions", use_container_width=True):
            st.switch_page("pages/3_Sessions.py")
        
        if st.button("🎭 Play", key="side_play", use_container_width=True):
            st.switch_page("pages/4_Play.py")
        
        st.divider()
        
        # Quick stats
        db = get_database()
        
        st.markdown("### 📊 Stats")
        st.caption(f"📚 {db.get_rulebook_count()} rulebooks indexed")
        st.caption(f"💾 {db.get_session_count()} sessions saved")
        
        st.divider()
        
        st.markdown("### About")
        st.markdown("""
        DungeonAI uses a **neuro-symbolic** architecture:
        
        - **Python** handles truth (state, dice, rules)
        - **AI** handles narrative and reasoning
        
        The AI cannot fudge dice or invent stats.
        """)


# =============================================================================
# Entry Point
# =============================================================================


render_sidebar()
main()
