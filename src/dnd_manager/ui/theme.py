"""DungeonAI Theme - Modern D&D Aesthetic.

A clean, readable theme inspired by modern fantasy games and D&D Beyond.
Prioritizes readability and professional appearance.
"""

from __future__ import annotations

import streamlit as st


# =============================================================================
# Color Palette - High Contrast, Modern Fantasy
# =============================================================================


class Colors:
    """Modern D&D color palette with proper contrast."""
    
    # Primary accent
    CRIMSON = "#DC2626"
    CRIMSON_DARK = "#B91C1C"
    
    # Gold accents (softer, less yellow)
    AMBER = "#D97706"
    AMBER_LIGHT = "#F59E0B"
    
    # Backgrounds - clean and readable
    BG_DARK = "#0F0F0F"
    BG_CARD = "#1A1A1A"
    BG_ELEVATED = "#242424"
    BG_HOVER = "#2D2D2D"
    
    # Text - high contrast
    TEXT_PRIMARY = "#FAFAFA"
    TEXT_SECONDARY = "#A1A1A1"
    TEXT_MUTED = "#737373"
    TEXT_DARK = "#171717"
    
    # Borders
    BORDER = "#333333"
    BORDER_LIGHT = "#404040"
    
    # Status
    SUCCESS = "#22C55E"
    WARNING = "#F59E0B"
    ERROR = "#EF4444"
    INFO = "#3B82F6"
    
    # HP colors
    HP_HEALTHY = "#22C55E"
    HP_WOUNDED = "#F59E0B"
    HP_CRITICAL = "#EF4444"


# =============================================================================
# Main CSS - Dark Mode, High Contrast
# =============================================================================


THEME_CSS = """
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Cinzel:wght@500;600;700&display=swap');
    
    /* =================================
       CSS VARIABLES
       ================================= */
    :root {
        --bg-dark: #1C1410;
        --bg-card: #2A201A;
        --bg-elevated: #3D2E24;
        --bg-hover: #4A3828;
        --text-primary: #F5EDE4;
        --text-secondary: #C4B5A5;
        --text-muted: #8B7355;
        --crimson: #8B2020;
        --crimson-dark: #6B1818;
        --amber: #C9A227;
        --border: #5C4A3A;
        --border-light: #705840;
        --font-display: 'Cinzel', serif;
        --font-body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* =================================
       BASE STYLES
       ================================= */
    
    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Header - minimal */
    header[data-testid="stHeader"] {
        background: var(--bg-dark);
        border-bottom: 1px solid var(--border);
    }
    
    /* Main content */
    .main {
        background: var(--bg-dark);
    }
    
    .main .block-container {
        padding: 2rem;
        max-width: 1200px;
        font-family: var(--font-body);
        color: var(--text-primary);
    }
    
    /* =================================
       TYPOGRAPHY
       ================================= */
    
    h1, h2, h3 {
        font-family: var(--font-display) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }
    
    h1 {
        font-size: 2rem !important;
        padding-bottom: 0.75rem;
        margin-bottom: 1.5rem !important;
        border-bottom: 2px solid var(--crimson);
    }
    
    h2 {
        font-size: 1.5rem !important;
        color: var(--text-primary) !important;
        margin-top: 1.5rem !important;
    }
    
    h3 {
        font-size: 1.125rem !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    
    p, li, span, div, label {
        font-family: var(--font-body);
        color: var(--text-primary);
        line-height: 1.6;
    }
    
    /* Prevent iOS zoom */
    input, textarea, select {
        font-size: 16px !important;
    }
    
    /* =================================
       SIDEBAR
       ================================= */
    
    [data-testid="stSidebar"] {
        background: var(--bg-card);
        border-right: 1px solid var(--border);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text-primary);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: var(--text-secondary) !important;
    }
    
    /* Sidebar metrics */
    [data-testid="stSidebar"] .stMetric {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    [data-testid="stSidebar"] .stMetric label {
        color: var(--text-muted) !important;
    }
    
    [data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }
    
    /* =================================
       BUTTONS
       ================================= */
    
    .stButton > button {
        font-family: var(--font-body);
        font-weight: 500;
        background: var(--crimson);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        transition: all 0.15s ease;
    }
    
    .stButton > button:hover {
        background: var(--crimson-dark);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Secondary buttons (sidebar) */
    [data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: 1px solid var(--border-light);
        color: var(--text-primary);
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--bg-hover);
        border-color: var(--crimson);
        box-shadow: none;
    }
    
    /* =================================
       CARDS & CONTAINERS
       ================================= */
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
    }
    
    [data-testid="stMetric"] label {
        color: var(--text-muted) !important;
        font-size: 0.875rem;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: var(--font-display);
        font-size: 1.5rem;
    }
    
    /* =================================
       CHARACTER SHEET COMPONENTS
       ================================= */
    
    .stat-block {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-top: 3px solid var(--crimson);
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
    }
    
    .stat-block-header {
        font-family: var(--font-display);
        font-size: 1.25rem;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .stat-block-subheader {
        font-style: italic;
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin-bottom: 0.75rem;
    }
    
    .stat-block hr {
        border: none;
        height: 1px;
        background: var(--border);
        margin: 0.75rem 0;
    }
    
    /* Ability score grid */
    .ability-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .ability-box {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.5rem;
        text-align: center;
    }
    
    .ability-label {
        font-size: 0.625rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .ability-score {
        font-family: var(--font-display);
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .ability-mod {
        font-size: 0.875rem;
        color: var(--amber);
        font-weight: 500;
    }
    
    /* HP Bar */
    .hp-container {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }
    
    .hp-bar {
        height: 20px;
        background: var(--bg-elevated);
        border-radius: 10px;
        overflow: hidden;
        position: relative;
    }
    
    .hp-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .hp-fill.healthy { background: linear-gradient(90deg, #3D6B35, #4A7C3F); }
    .hp-fill.wounded { background: linear-gradient(90deg, #A68520, #C9A227); }
    .hp-fill.critical { background: linear-gradient(90deg, #6B1818, #8B2020); }
    
    .hp-text {
        position: absolute;
        width: 100%;
        text-align: center;
        line-height: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }
    
    /* =================================
       CHAT MESSAGES
       ================================= */
    
    .chat-dm {
        background: var(--bg-card);
        border-left: 3px solid var(--crimson);
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    
    .chat-player {
        background: var(--bg-elevated);
        border-right: 3px solid var(--amber);
        border-radius: 8px 0 0 8px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        margin-left: 10%;
    }
    
    .chat-system {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.875rem;
        color: var(--text-secondary);
        text-align: center;
    }
    
    .chat-roll {
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
        border: 1px solid var(--amber);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.75rem 0;
        font-family: 'SF Mono', 'Fira Code', monospace;
        color: var(--amber);
        text-align: center;
    }
    
    .chat-roll .dice-result {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    /* =================================
       DICE ROLLING ANIMATION
       ================================= */
    
    @keyframes dice-roll {
        0% { transform: rotate(0deg) scale(1); }
        25% { transform: rotate(90deg) scale(1.1); }
        50% { transform: rotate(180deg) scale(1); }
        75% { transform: rotate(270deg) scale(1.1); }
        100% { transform: rotate(360deg) scale(1); }
    }
    
    @keyframes dice-bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes dice-glow {
        0%, 100% { filter: drop-shadow(0 0 0px var(--amber)); }
        50% { filter: drop-shadow(0 0 8px var(--amber)); }
    }
    
    .dice-rolling {
        display: inline-block;
        animation: dice-roll 0.6s ease-out, dice-bounce 0.6s ease-out, dice-glow 0.6s ease-out;
    }
    
    .dice-natural-20 {
        display: inline-block;
        animation: dice-roll 0.6s ease-out, dice-glow 1.5s ease-in-out infinite;
        color: var(--crimson) !important;
        text-shadow: 0 0 10px var(--crimson);
    }
    
    .dice-natural-1 {
        display: inline-block;
        animation: dice-roll 0.6s ease-out;
        opacity: 0.6;
    }
    
    /* =================================
       SPELL SLOTS
       ================================= */
    
    .spell-slots {
        display: flex;
        gap: 4px;
        flex-wrap: wrap;
    }
    
    .spell-slot {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        border: 2px solid var(--crimson);
    }
    
    .spell-slot.available {
        background: var(--crimson);
    }
    
    .spell-slot.used {
        background: transparent;
    }
    
    /* =================================
       FORMS & INPUTS
       ================================= */
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body);
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--crimson) !important;
        box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.2) !important;
    }
    
    .stTextInput > label,
    .stTextArea > label,
    .stSelectbox > label {
        color: var(--text-secondary) !important;
    }
    
    .stSelectbox > div > div {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: var(--bg-elevated);
        border: 2px dashed var(--border);
        border-radius: 8px;
        padding: 1rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--crimson);
    }
    
    [data-testid="stFileUploader"] label {
        color: var(--text-secondary) !important;
    }
    
    /* =================================
       EXPANDERS
       ================================= */
    
    .streamlit-expanderHeader {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
    }
    
    /* =================================
       TABS
       ================================= */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 1px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-body);
        color: var(--text-secondary);
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--text-primary);
        border-bottom-color: var(--crimson);
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding: 1rem 0;
    }
    
    /* =================================
       DIVIDERS
       ================================= */
    
    hr {
        border: none;
        height: 1px;
        background: var(--border);
        margin: 1.5rem 0;
    }
    
    /* =================================
       INFO/WARNING/ERROR
       ================================= */
    
    .stAlert {
        background: var(--bg-elevated) !important;
        border-radius: 6px !important;
    }
    
    [data-testid="stAlert"] > div {
        color: var(--text-primary) !important;
    }
    
    /* =================================
       CHAT INPUT
       ================================= */
    
    [data-testid="stChatInput"] {
        background: var(--bg-card) !important;
        border-top: 1px solid var(--border) !important;
    }
    
    [data-testid="stChatInput"] textarea {
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }
    
    /* =================================
       CHAT MESSAGES (Native)
       ================================= */
    
    [data-testid="stChatMessage"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    
    /* =================================
       MOBILE
       ================================= */
    
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.25rem !important; }
        
        .ability-grid {
            grid-template-columns: repeat(3, 1fr);
        }
        
        .chat-player {
            margin-left: 5%;
        }
        
        .stButton > button {
            min-height: 44px;
        }
    }
    
    /* =================================
       SCROLLBAR
       ================================= */
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-light);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
</style>
"""


# =============================================================================
# Theme Application
# =============================================================================


def apply_theme() -> None:
    """Apply the theme to the Streamlit app."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_stat_block(
    name: str,
    subtitle: str,
    stats: dict[str, int],
    ac: int,
    hp: int,
    speed: str = "30 ft.",
    abilities: list[str] | None = None,
    actions: list[dict[str, str]] | None = None,
) -> None:
    """Render a D&D-style stat block."""
    def mod(score: int) -> str:
        m = (score - 10) // 2
        return f"+{m}" if m >= 0 else str(m)
    
    abilities_html = ""
    if abilities:
        abilities_html = "<hr>" + "".join(
            f"<p><strong>{a.split(':')[0]}:</strong> {':'.join(a.split(':')[1:])}</p>"
            for a in abilities
        )
    
    actions_html = ""
    if actions:
        actions_html = "<hr><p><strong>Actions</strong></p>" + "".join(
            f"<p><em><strong>{a['name']}.</strong></em> {a['desc']}</p>"
            for a in actions
        )
    
    st.markdown(f"""
    <div class="stat-block">
        <div class="stat-block-header">{name}</div>
        <div class="stat-block-subheader">{subtitle}</div>
        <hr>
        <p><strong>Armor Class</strong> {ac}</p>
        <p><strong>Hit Points</strong> {hp}</p>
        <p><strong>Speed</strong> {speed}</p>
        <hr>
        <table style="width:100%; text-align:center; border-collapse:collapse; color: var(--text-primary);">
            <tr style="color: var(--text-muted); font-size: 0.75rem;">
                <th>STR</th><th>DEX</th><th>CON</th><th>INT</th><th>WIS</th><th>CHA</th>
            </tr>
            <tr>
                <td>{stats.get('strength', 10)} ({mod(stats.get('strength', 10))})</td>
                <td>{stats.get('dexterity', 10)} ({mod(stats.get('dexterity', 10))})</td>
                <td>{stats.get('constitution', 10)} ({mod(stats.get('constitution', 10))})</td>
                <td>{stats.get('intelligence', 10)} ({mod(stats.get('intelligence', 10))})</td>
                <td>{stats.get('wisdom', 10)} ({mod(stats.get('wisdom', 10))})</td>
                <td>{stats.get('charisma', 10)} ({mod(stats.get('charisma', 10))})</td>
            </tr>
        </table>
        {abilities_html}
        {actions_html}
    </div>
    """, unsafe_allow_html=True)


def render_hp_bar(current: int, maximum: int, temp: int = 0) -> None:
    """Render an HP bar with color coding."""
    pct = (current / maximum * 100) if maximum > 0 else 0
    hp_class = "healthy" if pct > 50 else ("wounded" if pct > 25 else "critical")
    
    temp_display = f" (+{temp})" if temp > 0 else ""
    
    st.markdown(f"""
    <div class="hp-container">
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-weight: 600; color: var(--text-primary);">Hit Points</span>
            <span style="color: var(--text-secondary);">{current}/{maximum}{temp_display}</span>
        </div>
        <div class="hp-bar">
            <div class="hp-fill {hp_class}" style="width: {pct}%;"></div>
            <span class="hp-text">{current}/{maximum}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_ability_scores(stats: dict[str, int]) -> None:
    """Render ability scores in a grid."""
    def mod(score: int) -> str:
        m = (score - 10) // 2
        return f"+{m}" if m >= 0 else str(m)
    
    abilities = [
        ("STR", stats.get("strength", 10)),
        ("DEX", stats.get("dexterity", 10)),
        ("CON", stats.get("constitution", 10)),
        ("INT", stats.get("intelligence", 10)),
        ("WIS", stats.get("wisdom", 10)),
        ("CHA", stats.get("charisma", 10)),
    ]
    
    boxes = "".join(f"""
        <div class="ability-box">
            <div class="ability-label">{name}</div>
            <div class="ability-score">{score}</div>
            <div class="ability-mod">{mod(score)}</div>
        </div>
    """ for name, score in abilities)
    
    st.markdown(f'<div class="ability-grid">{boxes}</div>', unsafe_allow_html=True)


def render_chat_message(role: str, content: str) -> None:
    """Render a styled chat message."""
    css_class = {
        "dm": "chat-dm",
        "player": "chat-player",
        "system": "chat-system",
        "roll": "chat-roll",
    }.get(role, "chat-system")
    
    st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)


def render_dice_result(
    result: int | str,
    is_natural_20: bool = False,
    is_natural_1: bool = False,
    animate: bool = True,
) -> None:
    """Render a dice roll result with optional animation.
    
    Args:
        result: The dice roll result (number or formatted string).
        is_natural_20: Whether this is a natural 20 (critical hit).
        is_natural_1: Whether this is a natural 1 (fumble).
        animate: Whether to animate the dice emoji.
    """
    dice_emoji = "🎲"
    
    if not animate:
        st.markdown(f"{dice_emoji} **{result}**")
        return
    
    # Determine animation class
    if is_natural_20:
        css_class = "dice-natural-20"
        result_text = f"<span style='color: var(--crimson); font-weight: 700;'>⚔️ **CRITICAL!** {result}</span>"
    elif is_natural_1:
        css_class = "dice-natural-1"
        result_text = f"<span style='opacity: 0.6;'>💥 **FUMBLE!** {result}</span>"
    else:
        css_class = "dice-rolling"
        result_text = f"**{result}**"
    
    st.markdown(
        f'<span class="{css_class}">{dice_emoji}</span> {result_text}',
        unsafe_allow_html=True,
    )


def render_spell_slots(slots: dict[int, tuple[int, int]]) -> None:
    """Render spell slot indicators."""
    for level, (current, max_val) in sorted(slots.items()):
        dots = "".join(
            f'<span class="spell-slot {"available" if i < current else "used"}"></span>'
            for i in range(max_val)
        )
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin: 6px 0;">
            <span style="width: 60px; font-weight: 500; color: var(--text-secondary);">Level {level}</span>
            <div class="spell-slots">{dots}</div>
            <span style="color: var(--text-muted); font-size: 0.875rem;">({current}/{max_val})</span>
        </div>
        """, unsafe_allow_html=True)
