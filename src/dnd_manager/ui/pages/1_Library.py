"""Library Page - Manage Rulebooks and Expansions.

This page allows users to:
- Upload and index new rulebooks/expansions
- View all indexed documents
- Remove documents from the library
- Documents persist across sessions via SQLite
"""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from dnd_manager.core.logging import get_logger
from dnd_manager.ingestion.universal_loader import (
    ChromaStore,
    DocumentType,
    UniversalIngestor,
)
from dnd_manager.storage.database import RulebookRecord, get_database
from dnd_manager.ui.theme import Colors, apply_theme

logger = get_logger(__name__)


# =============================================================================
# Page Configuration
# =============================================================================


st.set_page_config(
    page_title="Library | DungeonAI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()


# =============================================================================
# Session State
# =============================================================================


def init_session_state() -> None:
    """Initialize session state for the library page."""
    if "ingestor" not in st.session_state:
        # Use persistent ChromaDB location
        chroma_path = Path.home() / ".dungeonai" / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        chroma_store = ChromaStore(persist_directory=str(chroma_path))
        st.session_state.ingestor = UniversalIngestor(chroma_store=chroma_store)
    
    if "upload_success" not in st.session_state:
        st.session_state.upload_success = None


def get_ingestor() -> UniversalIngestor:
    """Get the universal ingestor instance."""
    init_session_state()
    return st.session_state.ingestor


# =============================================================================
# Helper Functions
# =============================================================================


def calculate_file_hash(file_bytes: bytes) -> str:
    """Calculate SHA256 hash of file contents."""
    return hashlib.sha256(file_bytes).hexdigest()


def format_date(dt: datetime) -> str:
    """Format datetime for display."""
    return dt.strftime("%b %d, %Y at %I:%M %p")


def get_doc_type_icon(doc_type: str) -> str:
    """Get icon for document type."""
    icons = {
        "sourcebook": "📖",
        "adventure": "🗺️",
        "third_party": "🎲",
        "character_sheet": "📋",
    }
    return icons.get(doc_type, "📄")


# =============================================================================
# Main Page
# =============================================================================


def main() -> None:
    """Render the library page."""
    init_session_state()
    db = get_database()
    
    # Header
    st.title("📚 Library")
    st.markdown("""
    Manage your D&D rulebooks, expansions, and reference materials. 
    Documents indexed here are available across all your game sessions.
    """)
    
    # Show success message if any
    if st.session_state.upload_success:
        st.success(st.session_state.upload_success)
        st.session_state.upload_success = None
    
    # Layout: Upload section and Library section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        render_upload_section()
    
    with col2:
        render_library_section()


def render_upload_section() -> None:
    """Render the document upload section."""
    st.markdown("### Add to Library")
    
    # Document type selection
    doc_type = st.selectbox(
        "Document Type",
        options=[
            ("Sourcebook (PHB, XGE, MM, Tasha's, etc.)", "sourcebook"),
            ("Adventure (Strahd, Dragon Heist, etc.)", "adventure"),
            ("Third Party / Homebrew", "third_party"),
        ],
        format_func=lambda x: x[0],
        key="doc_type_select",
    )
    
    # Custom name (optional)
    custom_name = st.text_input(
        "Display Name (optional)",
        placeholder="Leave blank to use filename",
        key="custom_name_input",
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="library_upload",
        help="Upload a D&D PDF to index for RAG retrieval",
    )
    
    # Upload button
    if uploaded_file:
        if st.button("📥 Index Document", key="index_btn", use_container_width=True):
            process_upload(uploaded_file, doc_type[1], custom_name)


def process_upload(uploaded_file, doc_type: str, custom_name: str) -> None:
    """Process and index an uploaded document."""
    db = get_database()
    ingestor = get_ingestor()
    
    # Read file bytes
    file_bytes = uploaded_file.read()
    file_hash = calculate_file_hash(file_bytes)
    
    # Check for duplicates
    existing = db.get_rulebook_by_hash(file_hash)
    if existing:
        st.warning(f"This document is already indexed as **{existing.name}**")
        return
    
    # Determine display name
    display_name = custom_name.strip() if custom_name.strip() else uploaded_file.name
    
    with st.spinner(f"Indexing **{display_name}**... This may take a minute."):
        try:
            # Save to temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            
            # Index based on document type
            if doc_type == "adventure":
                chunk_count = ingestor.ingest_adventure_module(tmp_path, display_name)
            else:
                chunk_count = ingestor.ingest_rulebook(tmp_path, display_name)
            
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
            
            # Save to database
            db.add_rulebook(
                name=display_name,
                filename=uploaded_file.name,
                doc_type=doc_type,
                chunk_count=chunk_count,
                file_hash=file_hash,
            )
            
            st.session_state.upload_success = f"Successfully indexed **{display_name}** ({chunk_count} chunks)"
            st.rerun()
            
        except Exception as e:
            logger.exception("Failed to index document")
            st.error(f"Failed to index document: {e}")


def render_library_section() -> None:
    """Render the library contents section."""
    st.markdown("### Your Library")
    
    db = get_database()
    rulebooks = db.get_all_rulebooks()
    
    if not rulebooks:
        st.info("""
        Your library is empty. Upload rulebooks and expansions to get started.
        
        **Recommended documents:**
        - Player's Handbook (PHB)
        - Dungeon Master's Guide (DMG)  
        - Monster Manual (MM)
        - Adventure modules you want to run
        """)
        return
    
    # Summary stats
    total_chunks = sum(r.chunk_count for r in rulebooks)
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(rulebooks))
    col2.metric("Total Chunks", f"{total_chunks:,}")
    col3.metric("Types", len(set(r.doc_type for r in rulebooks)))
    
    st.divider()
    
    # Group by document type
    by_type: dict[str, list[RulebookRecord]] = {}
    for rulebook in rulebooks:
        if rulebook.doc_type not in by_type:
            by_type[rulebook.doc_type] = []
        by_type[rulebook.doc_type].append(rulebook)
    
    # Render each type
    for doc_type, docs in sorted(by_type.items()):
        icon = get_doc_type_icon(doc_type)
        type_name = doc_type.replace("_", " ").title()
        
        with st.expander(f"{icon} {type_name} ({len(docs)})", expanded=True):
            for doc in docs:
                render_rulebook_card(doc)


def render_rulebook_card(rulebook: RulebookRecord) -> None:
    """Render a single rulebook card."""
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"**{rulebook.name}**")
        st.caption(f"📄 {rulebook.filename}")
    
    with col2:
        st.markdown(f"**{rulebook.chunk_count}** chunks")
        st.caption(format_date(rulebook.indexed_at))
    
    with col3:
        if st.button("🗑️ Remove", key=f"remove_{rulebook.id}", use_container_width=True):
            remove_rulebook(rulebook)


def remove_rulebook(rulebook: RulebookRecord) -> None:
    """Remove a rulebook from the library."""
    db = get_database()
    
    # TODO: Also remove from ChromaDB
    # This would require tracking chunk IDs or using a source filter
    
    if db.delete_rulebook(rulebook.id):
        st.session_state.upload_success = f"Removed **{rulebook.name}** from library"
        st.rerun()
    else:
        st.error("Failed to remove document")


# =============================================================================
# Sidebar
# =============================================================================


def render_sidebar() -> None:
    """Render the sidebar with navigation and help."""
    with st.sidebar:
        st.markdown("# 📚 Library")
        st.markdown("Manage your D&D reference materials.")
        
        st.divider()
        
        # Quick stats
        db = get_database()
        rulebooks = db.get_all_rulebooks()
        sessions = db.get_session_count()
        
        st.markdown("### 📊 Stats")
        st.metric("Documents", len(rulebooks))
        st.metric("Saved Sessions", sessions)
        
        st.divider()
        
        # Help section
        st.markdown("### ❓ Help")
        st.markdown("""
        **What can I upload?**
        - Official rulebooks (PHB, DMG, etc.)
        - Third-party expansions
        - Adventure modules
        - Monster compendiums
        
        **How does it work?**
        Documents are converted to searchable chunks 
        that the AI DM can reference during gameplay.
        
        **File format**
        Currently only PDF files are supported.
        """)


# =============================================================================
# Entry Point
# =============================================================================


render_sidebar()
main()
