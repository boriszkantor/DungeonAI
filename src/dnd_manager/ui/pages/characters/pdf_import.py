"""PDF character sheet import functionality."""

from __future__ import annotations

import streamlit as st

from dnd_manager.core.logging import get_logger

from .state import save_characters

logger = get_logger(__name__)


def render_pdf_upload() -> None:
    """Render PDF upload section for character sheet extraction."""
    st.markdown("### 📥 Import from PDF")
    
    uploaded_file = st.file_uploader(
        "Upload Character Sheet",
        type=["pdf"],
        key="char_pdf_upload",
        help="Upload a D&D character sheet PDF to extract",
    )
    
    if uploaded_file:
        if st.button("🔮 Extract Character", key="extract_btn"):
            with st.spinner("Reading character sheet with AI vision..."):
                try:
                    ingestor = st.session_state.ingestor
                    pdf_bytes = uploaded_file.read()
                    
                    character = ingestor.ingest_character_sheet(pdf_bytes)
                    
                    # Save to characters
                    st.session_state.characters[str(character.uid)] = character
                    save_characters()
                    
                    st.session_state.char_message = ("success", f"Imported **{character.name}**!")
                    st.rerun()
                    
                except Exception as e:
                    logger.exception("Failed to extract character")
                    st.error(f"Extraction failed: {e}")
