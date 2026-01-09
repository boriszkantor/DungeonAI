"""Settings page for DungeonAI configuration.

Allows users to customize:
- API provider and key
- Model selection (dynamically fetched from OpenRouter)
- Other preferences
"""

from __future__ import annotations

import os
import requests
import streamlit as st
from pathlib import Path

from dnd_manager.ui.theme import apply_theme

st.set_page_config(
    page_title="Settings - DungeonAI",
    page_icon="⚙️",
    layout="wide",
)

apply_theme()

# Fallback models if API fetch fails
FALLBACK_MODELS = [
    {"id": "google/gemini-2.5-pro-preview", "name": "Gemini 2.5 Pro", "context_length": 1000000, "pricing": {"prompt": "1.25", "completion": "5.00"}, "description": "Google's best reasoning model"},
    {"id": "google/gemini-2.5-flash-preview", "name": "Gemini 2.5 Flash", "context_length": 1000000, "pricing": {"prompt": "0.15", "completion": "0.60"}, "description": "Fast Gemini with great tool use"},
    {"id": "google/gemini-2.0-flash-001", "name": "Gemini 2.0 Flash", "context_length": 1000000, "pricing": {"prompt": "0.10", "completion": "0.40"}, "description": "Fast and efficient"},
    {"id": "anthropic/claude-sonnet-4", "name": "Claude Sonnet 4", "context_length": 200000, "pricing": {"prompt": "3.00", "completion": "15.00"}, "description": "Excellent for creative writing and tool use"},
    {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet", "context_length": 200000, "pricing": {"prompt": "3.00", "completion": "15.00"}, "description": "Great balance of quality and speed"},
    {"id": "anthropic/claude-3-haiku", "name": "Claude 3 Haiku", "context_length": 200000, "pricing": {"prompt": "0.25", "completion": "1.25"}, "description": "Fast and affordable"},
    {"id": "openai/gpt-4o", "name": "GPT-4o", "context_length": 128000, "pricing": {"prompt": "2.50", "completion": "10.00"}, "description": "OpenAI's flagship model"},
    {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "context_length": 128000, "pricing": {"prompt": "0.15", "completion": "0.60"}, "description": "Smaller, faster GPT-4o"},
    {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B", "context_length": 131072, "pricing": {"prompt": "0.40", "completion": "0.40"}, "description": "Open source from Meta"},
    {"id": "deepseek/deepseek-chat", "name": "DeepSeek Chat", "context_length": 64000, "pricing": {"prompt": "0.14", "completion": "0.28"}, "description": "Very affordable"},
]



def fetch_openrouter_models_direct() -> tuple[list[dict], str | None]:
    """Fetch available models from OpenRouter API (no caching).
    
    Returns:
        Tuple of (models list, error message or None)
    """
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        
        models = []
        for model in data.get("data", []):
            # Extract relevant info
            try:
                prompt_price = float(model.get("pricing", {}).get("prompt", 0)) * 1_000_000
                completion_price = float(model.get("pricing", {}).get("completion", 0)) * 1_000_000
            except (ValueError, TypeError):
                prompt_price = 0
                completion_price = 0
            
            model_info = {
                "id": model.get("id", ""),
                "name": model.get("name", model.get("id", "Unknown")),
                "description": model.get("description", ""),
                "context_length": model.get("context_length", 0),
                "pricing": {
                    "prompt": str(prompt_price),
                    "completion": str(completion_price),
                },
            }
            if model_info["id"]:  # Only add if has valid ID
                models.append(model_info)
        
        # Sort by provider then name
        models.sort(key=lambda m: (m["id"].split("/")[0], m["name"]))
        return models, None
        
    except Exception as e:
        import traceback
        return FALLBACK_MODELS, f"{e}\n{traceback.format_exc()}"


def get_cached_models() -> tuple[list[dict], str | None]:
    """Get models with simple session-based caching."""
    # Check if we need to fetch
    if "models_cache" not in st.session_state:
        st.session_state.models_cache = None
        st.session_state.models_error = None
    
    if st.session_state.models_cache is None:
        models, error = fetch_openrouter_models_direct()
        st.session_state.models_cache = models
        st.session_state.models_error = error
    
    return st.session_state.models_cache, st.session_state.models_error


def get_model_display_name(model: dict) -> str:
    """Format model for display in dropdown."""
    name = model.get("name", model.get("id", "Unknown"))
    
    # Get pricing
    try:
        prompt_price = float(model.get("pricing", {}).get("prompt", 0))
        if prompt_price < 0.01:
            price_str = "free"
        elif prompt_price < 1:
            price_str = f"${prompt_price:.2f}/M"
        else:
            price_str = f"${prompt_price:.1f}/M"
    except (ValueError, TypeError):
        price_str = "?"
    
    # Get context length
    ctx = model.get("context_length", 0)
    if ctx >= 1_000_000:
        ctx_str = f"{ctx // 1_000_000}M ctx"
    elif ctx >= 1000:
        ctx_str = f"{ctx // 1000}K ctx"
    else:
        ctx_str = ""
    
    return f"{name} ({price_str}, {ctx_str})"


def load_settings_from_env() -> dict[str, str]:
    """Load settings from .env file or environment."""
    settings = {
        "openrouter_api_key": "",
        "dm_model": "",
        "vision_model": "",
    }
    
    # First try environment variables
    settings["openrouter_api_key"] = os.environ.get("DND_MANAGER_OPENROUTER_API_KEY", "")
    settings["dm_model"] = os.environ.get("DND_MANAGER_DM_MODEL", "")
    settings["vision_model"] = os.environ.get("DND_MANAGER_VISION_MODEL", "")
    
    # Try reading .env file directly for any missing values
    env_path = Path(".env")
    if env_path.exists():
        try:
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == "DND_MANAGER_OPENROUTER_API_KEY" and not settings["openrouter_api_key"]:
                        settings["openrouter_api_key"] = value
                    elif key == "DND_MANAGER_DM_MODEL" and not settings["dm_model"]:
                        settings["dm_model"] = value
                    elif key == "DND_MANAGER_VISION_MODEL" and not settings["vision_model"]:
                        settings["vision_model"] = value
        except Exception:
            pass
    
    return settings


def init_settings_state() -> None:
    """Initialize settings in session state, loading from .env on first run."""
    if "settings" not in st.session_state:
        # Load saved settings from .env
        loaded = load_settings_from_env()
        
        st.session_state.settings = {
            "api_provider": "openrouter",
            "openrouter_api_key": loaded["openrouter_api_key"] or "",
            "dm_model": loaded["dm_model"] or "google/gemini-2.5-flash-preview",  # Good tool use support
            "vision_model": loaded["vision_model"] or "google/gemini-2.0-flash-001",
            "auto_save": True,
            "show_dice_details": True,
            "compact_combat": False,
        }


def save_setting_to_env(key: str, value: str) -> bool:
    """Save a single setting to .env file.
    
    Args:
        key: Environment variable name (e.g., "DND_MANAGER_DM_MODEL")
        value: Value to save
        
    Returns:
        True if saved successfully
    """
    env_path = Path(".env")
    
    # Read existing .env content
    existing_lines = []
    if env_path.exists():
        try:
            with open(env_path, "r") as f:
                existing_lines = [line.rstrip("\n") for line in f.readlines()]
        except Exception:
            pass
    
    # Update or add the key
    found = False
    new_lines = []
    
    for line in existing_lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    
    if not found:
        new_lines.append(f"{key}={value}")
    
    # Write back
    try:
        with open(env_path, "w") as f:
            f.write("\n".join(new_lines))
            f.write("\n")
        os.environ[key] = value
        return True
    except Exception:
        return False


def save_settings_to_env() -> bool:
    """Save API key to .env file."""
    settings = st.session_state.settings
    api_key = settings.get("openrouter_api_key", "")
    
    if not api_key:
        st.error("No API key to save")
        return False
    
    return save_setting_to_env("DND_MANAGER_OPENROUTER_API_KEY", api_key)


def render_model_details(model_info: dict) -> None:
    """Render model details in an expander."""
    with st.expander("Model Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            ctx = model_info.get('context_length', 0)
            if ctx >= 1_000_000:
                ctx_display = f"{ctx // 1_000_000}M"
            elif ctx >= 1000:
                ctx_display = f"{ctx // 1000:,}K"
            else:
                ctx_display = f"{ctx:,}"
            st.metric("Context Length", ctx_display)
        with col2:
            try:
                prompt_price = float(model_info.get("pricing", {}).get("prompt", 0))
                st.metric("Input Cost", f"${prompt_price:.2f}/M tokens")
            except (ValueError, TypeError):
                st.metric("Input Cost", "Unknown")
        with col3:
            try:
                comp_price = float(model_info.get("pricing", {}).get("completion", 0))
                st.metric("Output Cost", f"${comp_price:.2f}/M tokens")
            except (ValueError, TypeError):
                st.metric("Output Cost", "Unknown")
        
        if model_info.get("description"):
            desc = model_info["description"]
            st.caption(desc[:300] + "..." if len(desc) > 300 else desc)


def render_api_settings() -> None:
    """Render API configuration settings."""
    st.markdown("### API Configuration")
    
    settings = st.session_state.settings
    
    # API Provider
    st.markdown("**Provider**")
    st.info("DungeonAI uses [OpenRouter](https://openrouter.ai/) to access multiple AI models with a single API key.")
    
    # API Key input
    current_key = settings.get("openrouter_api_key", "")
    api_key = st.text_input(
        "OpenRouter API Key",
        value=current_key,
        type="password",
        help="Get your API key from https://openrouter.ai/keys",
        key="api_key_input",
    )
    
    # Update session state if changed
    if api_key != current_key:
        settings["openrouter_api_key"] = api_key
    
    # Show status
    if api_key:
        if api_key.startswith("sk-or-"):
            st.success("API key format looks valid")
        else:
            st.warning("API key should start with 'sk-or-'")
    else:
        st.warning("No API key configured. AI features won't work.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.link_button("Get API Key", "https://openrouter.ai/keys", use_container_width=True)
    
    with col2:
        if st.button("Save API Key", use_container_width=True, type="primary"):
            if save_settings_to_env():
                st.success("Saved! The key is now stored in .env")


def render_model_settings() -> None:
    """Render model selection settings."""
    st.markdown("### Model Selection")
    
    settings = st.session_state.settings
    
    # Fetch models from API
    all_models, fetch_error = get_cached_models()
    
    # Create lookup for all models
    all_models_lookup = {m["id"]: m for m in all_models}
    
    # Show refresh button and count
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Refresh", help="Refresh model list from OpenRouter"):
            st.session_state.models_cache = None  # Clear the cache
            st.session_state.models_error = None
            st.rerun()
    
    with col1:
        if fetch_error:
            st.error(f"API Error: {fetch_error}")
            st.caption(f"Showing {len(all_models)} fallback models")
        else:
            st.caption(f"{len(all_models)} models available")
    
    st.divider()
    
    # ===================
    # DM Model selection
    # ===================
    st.markdown("**Dungeon Master Model**")
    st.caption("The AI model that runs your D&D games. Better models provide better storytelling but cost more.")
    
    # All models available - user can search/scroll through them
    dm_models = list(all_models)
    
    current_dm = settings.get("dm_model", "google/gemini-2.5-pro-preview")
    dm_options = [m["id"] for m in dm_models]
    dm_lookup = {m["id"]: m for m in dm_models}
    
    # Ensure current selection is valid
    if current_dm not in dm_options and dm_options:
        current_dm = dm_options[0]
        settings["dm_model"] = current_dm

    current_idx = dm_options.index(current_dm) if current_dm in dm_options else 0
    
    selected_dm = st.selectbox(
        "DM Model",
        options=dm_options,
        format_func=lambda x: get_model_display_name(dm_lookup.get(x, {"id": x, "name": x})),
        index=current_idx,
        key="dm_model_select",
        label_visibility="collapsed",
    )
    
    # Show selected model details
    if selected_dm in dm_lookup:
        render_model_details(dm_lookup[selected_dm])
    
    if selected_dm != settings.get("dm_model"):
        settings["dm_model"] = selected_dm
        # Clear DM so it recreates with new model
        if "dm" in st.session_state:
            st.session_state.dm = None
        # Save to .env for persistence
        save_setting_to_env("DND_MANAGER_DM_MODEL", selected_dm)
        st.success(f"DM model changed to: {selected_dm}")
    
    st.divider()
    
    # ======================
    # Vision Model selection
    # ======================
    st.markdown("**Vision Model**")
    st.caption("Used to extract data from uploaded PDFs (character sheets, rulebooks, adventures). Must support image input.")
    
    # All models available - not all support vision but user can choose
    vision_models = list(all_models)
    
    current_vision = settings.get("vision_model", "google/gemini-2.0-flash-001")
    vision_options = [m["id"] for m in vision_models]
    vision_lookup = {m["id"]: m for m in vision_models}
    
    # Ensure current selection is valid
    if current_vision not in vision_options and vision_options:
        current_vision = vision_options[0]
        settings["vision_model"] = current_vision
    
    current_vision_idx = vision_options.index(current_vision) if current_vision in vision_options else 0
    
    selected_vision = st.selectbox(
        "Vision Model",
        options=vision_options,
        format_func=lambda x: get_model_display_name(vision_lookup.get(x, {"id": x, "name": x})),
        index=current_vision_idx,
        key="vision_model_select",
        label_visibility="collapsed",
    )
    
    # Show selected vision model details
    if selected_vision in vision_lookup:
        render_model_details(vision_lookup[selected_vision])
    
    if selected_vision != settings.get("vision_model"):
        settings["vision_model"] = selected_vision
        # Save to .env for persistence
        save_setting_to_env("DND_MANAGER_VISION_MODEL", selected_vision)
        st.success(f"Vision model changed to: {selected_vision}")


def render_gameplay_settings() -> None:
    """Render gameplay preference settings."""
    st.markdown("### Gameplay Preferences")
    
    settings = st.session_state.settings
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_save = st.toggle(
            "Auto-save sessions",
            value=settings.get("auto_save", True),
            help="Automatically save game state after each action",
        )
        settings["auto_save"] = auto_save
        
        show_dice = st.toggle(
            "Show dice roll details",
            value=settings.get("show_dice_details", True),
            help="Show detailed dice roll breakdowns in chat",
        )
        settings["show_dice_details"] = show_dice
    
    with col2:
        compact_combat = st.toggle(
            "Compact combat display",
            value=settings.get("compact_combat", False),
            help="Use a more compact initiative tracker",
        )
        settings["compact_combat"] = compact_combat


def render_danger_zone() -> None:
    """Render dangerous actions (clear data, etc.)."""
    st.markdown("### Danger Zone")
    
    with st.expander("Clear Application Data", expanded=False):
        st.warning("These actions cannot be undone!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Clear Chat History", use_container_width=True):
                if "chat_history" in st.session_state:
                    st.session_state.chat_history = []
                st.success("Chat history cleared")
                st.rerun()
        
        with col2:
            if st.button("Clear Session", use_container_width=True):
                for key in ["game_state", "dm", "active_session_id", "chat_history"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Session cleared")
                st.rerun()
        
        with col3:
            if st.button("Reset All Settings", use_container_width=True):
                if "settings" in st.session_state:
                    del st.session_state["settings"]
                if "openrouter_models" in st.session_state:
                    del st.session_state["openrouter_models"]
                st.success("Settings reset to defaults")
                st.rerun()


def main() -> None:
    """Main settings page."""
    st.title("Settings")
    st.markdown("Configure DungeonAI to your preferences.")
    
    init_settings_state()
    
    # Create tabs for different settings sections
    tab1, tab2, tab3 = st.tabs(["API", "Models", "Gameplay"])
    
    with tab1:
        render_api_settings()
    
    with tab2:
        render_model_settings()
    
    with tab3:
        render_gameplay_settings()
        st.divider()
        render_danger_zone()
    
    # Footer
    st.divider()
    st.caption("Settings are stored in your browser session. API key is saved to .env file. Model changes take effect immediately.")


if __name__ == "__main__":
    main()
else:
    main()
