"""UI module for the D&D 5E AI Campaign Manager.

This module provides the Streamlit-based user interface for the
Campaign Manager, including the main app layout, components, and
theme styling.

Submodules:
    app: Main Streamlit application
    components: Reusable UI components
    theme: Visual styling and theming

Usage:
    Run the application with:
        streamlit run src/dnd_manager/ui/app.py

    Or import and run programmatically:
        from dnd_manager.ui import run_app
        run_app()
"""

from __future__ import annotations


def run_app() -> None:
    """Run the Streamlit application.
    
    This is a convenience function to start the app programmatically.
    Note: This launches a subprocess running streamlit.
    """
    import subprocess
    import sys
    from pathlib import Path
    
    app_path = Path(__file__).parent / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


__all__ = [
    "run_app",
]
