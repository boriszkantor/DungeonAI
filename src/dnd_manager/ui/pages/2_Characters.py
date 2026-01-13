"""Characters Page - Entry point that imports from modular structure.

This file serves as the Streamlit page entry point and imports
functionality from the characters/ submodule.
"""

from __future__ import annotations

from dnd_manager.ui.pages.characters import setup_and_run

# Execute the page setup and main function
setup_and_run()
