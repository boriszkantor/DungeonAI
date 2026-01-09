"""Characters Page - Manage Player Characters.

This page allows users to:
- Upload character sheet PDFs (vision extraction)
- Create new characters following D&D 5e rules
- View and manage saved characters
- Characters persist and can be used across sessions
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import streamlit as st

from dnd_manager.core.logging import get_logger
from dnd_manager.ingestion.universal_loader import ChromaStore, UniversalIngestor
from dnd_manager.models.ecs import (
    Ability,
    ActorEntity,
    ActorType,
    ClassFeatureComponent,
    ClassLevel,
    DefenseComponent,
    HealthComponent,
    InventoryComponent,
    ItemStack,
    ItemType,
    MovementComponent,
    SpellbookComponent,
    StatsComponent,
)
from dnd_manager.models.equipment import get_starting_equipment
from dnd_manager.storage.database import get_database
from dnd_manager.ui.theme import apply_theme, render_ability_scores, render_hp_bar

logger = get_logger(__name__)


# =============================================================================
# Page Configuration
# =============================================================================


st.set_page_config(
    page_title="Characters | DungeonAI",
    page_icon="🧙",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()


# =============================================================================
# Constants
# =============================================================================


RACES = [
    "Human", "Elf (High)", "Elf (Wood)", "Elf (Dark/Drow)", 
    "Dwarf (Hill)", "Dwarf (Mountain)", "Halfling (Lightfoot)", 
    "Halfling (Stout)", "Gnome (Forest)", "Gnome (Rock)",
    "Half-Elf", "Half-Orc", "Tiefling", "Dragonborn"
]

CLASSES = [
    "Barbarian", "Bard", "Cleric", "Druid", "Fighter",
    "Monk", "Paladin", "Ranger", "Rogue", "Sorcerer", 
    "Warlock", "Wizard"
]

BACKGROUNDS = [
    "Acolyte", "Charlatan", "Criminal", "Entertainer", "Folk Hero",
    "Guild Artisan", "Hermit", "Noble", "Outlander", "Sage",
    "Sailor", "Soldier", "Urchin"
]

ALIGNMENTS = [
    "Lawful Good", "Neutral Good", "Chaotic Good",
    "Lawful Neutral", "True Neutral", "Chaotic Neutral",
    "Lawful Evil", "Neutral Evil", "Chaotic Evil"
]

# Stat generation methods
STAT_METHODS = {
    "standard_array": "Standard Array (15, 14, 13, 12, 10, 8)",
    "point_buy": "Point Buy (27 points)",
    "roll": "Roll (4d6 drop lowest)",
    "manual": "Manual Entry",
}

STANDARD_ARRAY = [15, 14, 13, 12, 10, 8]

# Point buy costs
POINT_BUY_COSTS = {
    8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 7, 15: 9
}
POINT_BUY_TOTAL = 27

# Starting HP by class
CLASS_HIT_DIE = {
    "Barbarian": 12, "Bard": 8, "Cleric": 8, "Druid": 8,
    "Fighter": 10, "Monk": 8, "Paladin": 10, "Ranger": 10,
    "Rogue": 8, "Sorcerer": 6, "Warlock": 8, "Wizard": 6
}

# Saving throw proficiencies by class
CLASS_SAVE_PROFS = {
    "Barbarian": ["strength", "constitution"],
    "Bard": ["dexterity", "charisma"],
    "Cleric": ["wisdom", "charisma"],
    "Druid": ["intelligence", "wisdom"],
    "Fighter": ["strength", "constitution"],
    "Monk": ["strength", "dexterity"],
    "Paladin": ["wisdom", "charisma"],
    "Ranger": ["strength", "dexterity"],
    "Rogue": ["dexterity", "intelligence"],
    "Sorcerer": ["constitution", "charisma"],
    "Warlock": ["wisdom", "charisma"],
    "Wizard": ["intelligence", "wisdom"],
}


# =============================================================================
# Session State
# =============================================================================


def init_session_state() -> None:
    """Initialize session state."""
    if "characters" not in st.session_state:
        st.session_state.characters = load_characters()
    
    if "char_message" not in st.session_state:
        st.session_state.char_message = None
    
    if "ingestor" not in st.session_state:
        chroma_path = Path.home() / ".dungeonai" / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        chroma_store = ChromaStore(persist_directory=str(chroma_path))
        st.session_state.ingestor = UniversalIngestor(chroma_store=chroma_store)


def load_characters() -> dict[str, ActorEntity]:
    """Load saved characters from database."""
    db = get_database()
    characters = {}
    
    # Characters are stored in a special 'characters' session
    session = db.get_session("__characters__")
    if session:
        try:
            char_data = json.loads(session.game_state_json)
            for uid, data in char_data.items():
                characters[uid] = ActorEntity.model_validate(data)
        except Exception as e:
            logger.exception("Failed to load characters")
    
    return characters


def save_characters() -> None:
    """Save characters to database."""
    db = get_database()
    
    char_data = {
        uid: char.model_dump(mode="json")
        for uid, char in st.session_state.characters.items()
    }
    
    db.save_session(
        name="Character Storage",
        campaign_name="__system__",
        game_state_dict=char_data,
        chat_history=[],
        session_id="__characters__",
    )


def build_starting_inventory(class_name: str) -> InventoryComponent:
    """Build starting inventory for a character class.
    
    Args:
        class_name: Character class (e.g., 'Fighter', 'Wizard').
        
    Returns:
        InventoryComponent with starting equipment.
    """
    items, gold_cp = get_starting_equipment(class_name)
    
    inventory = InventoryComponent(
        items=items,
        currency_cp=gold_cp,
    )
    
    return inventory


def build_starting_spellbook(
    class_name: str,
    level: int,
    spellcasting_mod: int,
    proficiency_bonus: int,
) -> SpellbookComponent | None:
    """Build starting spellbook for a spellcaster class.
    
    Args:
        class_name: Character class.
        level: Character level.
        spellcasting_mod: Modifier for the spellcasting ability.
        proficiency_bonus: Character's proficiency bonus.
        
    Returns:
        SpellbookComponent or None if not a caster.
    """
    from dnd_manager.models.progression import (
        is_spellcaster,
        get_spellcasting_ability,
        get_spell_slots,
        get_starting_cantrips,
        get_starting_spells,
        get_cantrips_known,
        get_spells_known,
        get_pact_magic,
        FULL_CASTERS,
    )
    from dnd_manager.models.ecs import Ability
    
    if not is_spellcaster(class_name):
        return None
    
    # Get spellcasting ability
    ability_name = get_spellcasting_ability(class_name)
    ability_enum = Ability(ability_name) if ability_name else None
    
    # Calculate spell save DC and attack bonus
    spell_save_dc = 8 + proficiency_bonus + spellcasting_mod
    spell_attack_bonus = proficiency_bonus + spellcasting_mod
    
    # Get spell slots for this level
    slots_dict = get_spell_slots(class_name, level)
    # Convert to (current, max) tuples
    spell_slots = {lvl: (num, num) for lvl, num in slots_dict.items()}
    
    # Get starting cantrips (up to allowed amount)
    max_cantrips = get_cantrips_known(class_name, level)
    cantrips = get_starting_cantrips(class_name)[:max_cantrips]
    
    # Get starting spells (up to allowed amount)
    max_spells = get_spells_known(class_name, level)
    spells = get_starting_spells(class_name)
    
    # For prepared casters (Cleric, Druid, Paladin), spells_known is unlimited
    # but they prepare a subset. For now, give them the suggested list.
    if class_name in ("Cleric", "Druid", "Paladin"):
        # Prepared casters can prepare level + modifier spells
        max_prepared = max(1, level + spellcasting_mod)
        spells_known = spells[:max_prepared]
        spells_prepared = spells[:max_prepared]
    else:
        # Known casters (Bard, Sorcerer, Warlock, Wizard, Ranger)
        spells_known = spells[:max_spells] if max_spells > 0 else spells[:4]
        spells_prepared = []
    
    # Handle Warlock pact magic
    pact_slots_current = 0
    pact_slots_max = 0
    pact_slot_level = 0
    if class_name == "Warlock":
        pact_info = get_pact_magic(level)
        if pact_info:
            pact_slots_max, pact_slot_level = pact_info
            pact_slots_current = pact_slots_max
    
    return SpellbookComponent(
        spellcasting_ability=ability_enum,
        spell_save_dc=spell_save_dc,
        spell_attack_bonus=spell_attack_bonus,
        spell_slots=spell_slots,
        spells_known=spells_known,
        spells_prepared=spells_prepared,
        cantrips=cantrips,
        pact_slots_current=pact_slots_current,
        pact_slots_max=pact_slots_max,
        pact_slot_level=pact_slot_level,
    )


def calculate_ac_from_inventory(inventory: InventoryComponent, dex_mod: int) -> tuple[int, int | None, bool]:
    """Calculate AC from equipped armor.
    
    Args:
        inventory: Character's inventory.
        dex_mod: Character's DEX modifier.
        
    Returns:
        Tuple of (ac_base, max_dex_bonus, uses_dex).
    """
    ac_base = 10
    max_dex_bonus = None
    uses_dex = True
    shield_bonus = 0
    
    for item in inventory.items:
        if not item.equipped:
            continue
            
        if item.item_type == ItemType.ARMOR and item.ac_base is not None:
            ac_base = item.ac_base
            max_dex_bonus = item.max_dex_bonus
            uses_dex = max_dex_bonus is None or max_dex_bonus > 0
            
        elif item.item_type == ItemType.SHIELD and item.ac_bonus is not None:
            shield_bonus = item.ac_bonus
    
    return ac_base + shield_bonus, max_dex_bonus, uses_dex


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


def render_character_list() -> None:
    """Render the list of saved characters."""
    
    # Upload section
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


def render_character_editor(uid: str, character: ActorEntity) -> None:
    """Render a comprehensive editable character form covering all 5E sheet fields."""
    
    st.markdown("### ✏️ Edit Character")
    
    # Use tabs for organization
    tab_basic, tab_combat, tab_skills, tab_spells, tab_equipment, tab_personality, tab_appearance = st.tabs([
        "📋 Basic", "⚔️ Combat", "🎯 Skills", "✨ Spells", "🎒 Equipment", "💭 Personality", "👤 Appearance"
    ])
    
    # =========================================================================
    # TAB 1: Basic Information
    # =========================================================================
    with tab_basic:
        st.markdown("#### Basic Information")
        col1, col2 = st.columns(2)
        with col1:
            new_name = st.text_input("Name", value=character.name, key=f"edit_name_{uid}")
            new_race = st.text_input("Race", value=character.race, key=f"edit_race_{uid}")
            new_background = st.text_input("Background", value=character.background or "", key=f"edit_bg_{uid}")
        with col2:
            new_alignment = st.text_input("Alignment", value=character.alignment, key=f"edit_align_{uid}")
            new_size = st.selectbox(
                "Size",
                ["Tiny", "Small", "Medium", "Large", "Huge", "Gargantuan"],
                index=["Tiny", "Small", "Medium", "Large", "Huge", "Gargantuan"].index(character.size) if character.size in ["Tiny", "Small", "Medium", "Large", "Huge", "Gargantuan"] else 2,
                key=f"edit_size_{uid}"
            )
        
        st.divider()
        
        # Class & Level
        st.markdown("#### Class & Level")
        if character.class_features and character.class_features.classes:
            primary = character.class_features.classes[0]
            col1, col2, col3 = st.columns(3)
            new_class = col1.selectbox(
                "Class",
                CLASSES,
                index=CLASSES.index(primary.class_name) if primary.class_name in CLASSES else 0,
                key=f"edit_class_{uid}"
            )
            new_level = col2.number_input("Level", 1, 20, primary.level, key=f"edit_level_{uid}")
            new_subclass = col3.text_input("Subclass", value=primary.subclass or "", key=f"edit_subclass_{uid}")
        else:
            new_class = "Fighter"
            new_level = 1
            new_subclass = ""
        
        col1, col2 = st.columns(2)
        new_xp = col1.number_input("Experience Points", 0, 999999, character.experience_points, key=f"edit_xp_{uid}")
        new_prof_bonus = col2.number_input("Proficiency Bonus", 1, 10, character.stats.proficiency_bonus, key=f"edit_prof_{uid}")
        
        st.divider()
        
        # Ability Scores
        st.markdown("#### Ability Scores")
        cols = st.columns(6)
        new_str = cols[0].number_input("STR", 1, 30, character.stats.strength, key=f"edit_str_{uid}")
        new_dex = cols[1].number_input("DEX", 1, 30, character.stats.dexterity, key=f"edit_dex_{uid}")
        new_con = cols[2].number_input("CON", 1, 30, character.stats.constitution, key=f"edit_con_{uid}")
        new_int = cols[3].number_input("INT", 1, 30, character.stats.intelligence, key=f"edit_int_{uid}")
        new_wis = cols[4].number_input("WIS", 1, 30, character.stats.wisdom, key=f"edit_wis_{uid}")
        new_cha = cols[5].number_input("CHA", 1, 30, character.stats.charisma, key=f"edit_cha_{uid}")
    
    # =========================================================================
    # TAB 2: Combat Stats
    # =========================================================================
    with tab_combat:
        st.markdown("#### Hit Points")
        col1, col2, col3 = st.columns(3)
        new_hp_current = col1.number_input("Current HP", 0, 999, character.health.hp_current, key=f"edit_hp_cur_{uid}")
        new_hp_max = col2.number_input("Maximum HP", 1, 999, character.health.hp_max, key=f"edit_hp_max_{uid}")
        new_hp_temp = col3.number_input("Temp HP", 0, 999, character.health.hp_temp, key=f"edit_hp_temp_{uid}")
        
        st.divider()
        
        st.markdown("#### Armor Class")
        col1, col2, col3 = st.columns(3)
        new_ac_base = col1.number_input("AC Base", 0, 30, character.defense.ac_base, key=f"edit_ac_base_{uid}")
        new_ac_armor = col2.number_input("AC from Armor", 0, 20, character.defense.ac_armor, key=f"edit_ac_armor_{uid}")
        new_ac_shield = col3.number_input("AC from Shield", 0, 5, character.defense.ac_shield, key=f"edit_ac_shield_{uid}")
        
        st.divider()
        
        st.markdown("#### Movement")
        col1, col2, col3, col4 = st.columns(4)
        new_speed = col1.number_input("Walk (ft)", 0, 120, character.movement.speed_walk if character.movement else 30, key=f"edit_speed_{uid}")
        new_fly = col2.number_input("Fly (ft)", 0, 120, character.movement.speed_fly if character.movement else 0, key=f"edit_fly_{uid}")
        new_swim = col3.number_input("Swim (ft)", 0, 120, character.movement.speed_swim if character.movement else 0, key=f"edit_swim_{uid}")
        new_climb = col4.number_input("Climb (ft)", 0, 120, character.movement.speed_climb if character.movement else 0, key=f"edit_climb_{uid}")
        
        st.divider()
        
        st.markdown("#### Death Saves")
        col1, col2 = st.columns(2)
        new_death_success = col1.number_input("Successes", 0, 3, character.health.death_saves_success, key=f"edit_death_s_{uid}")
        new_death_fail = col2.number_input("Failures", 0, 3, character.health.death_saves_failure, key=f"edit_death_f_{uid}")
    
    # =========================================================================
    # TAB 3: Skills & Saves
    # =========================================================================
    with tab_skills:
        st.markdown("#### Saving Throw Proficiencies")
        st.caption("Check the saves your character is proficient in")
        
        save_abilities = ["strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"]
        current_saves = []
        for s in character.stats.save_proficiencies:
            if hasattr(s, 'value'):
                current_saves.append(s.value.lower())
            else:
                current_saves.append(str(s).lower())
        
        save_cols = st.columns(6)
        new_save_profs = []
        for i, ability in enumerate(save_abilities):
            if save_cols[i].checkbox(ability[:3].upper(), value=ability in current_saves, key=f"edit_save_{ability}_{uid}"):
                new_save_profs.append(ability)
        
        st.divider()
        
        st.markdown("#### Skill Proficiencies")
        st.caption("0 = Not proficient, 1 = Proficient, 2 = Expertise")
        
        from dnd_manager.models.progression import ALL_SKILLS, SKILL_ABILITIES
        
        # Display skills in 3 columns
        skill_cols = st.columns(3)
        new_skill_profs = {}
        
        for i, skill in enumerate(sorted(ALL_SKILLS)):
            col_idx = i % 3
            ability = SKILL_ABILITIES.get(skill, "dexterity")
            current_prof = character.stats.skill_proficiencies.get(skill, 0)
            
            skill_display = skill.replace("_", " ").title()
            new_val = skill_cols[col_idx].selectbox(
                f"{skill_display} ({ability[:3].upper()})",
                [0, 1, 2],
                index=current_prof if current_prof in [0, 1, 2] else 0,
                key=f"edit_skill_{skill}_{uid}",
                format_func=lambda x: {0: "—", 1: "★ Prof", 2: "★★ Expert"}[x]
            )
            if new_val > 0:
                new_skill_profs[skill] = new_val
    
    # =========================================================================
    # TAB 4: Spellcasting
    # =========================================================================
    with tab_spells:
        st.caption("💡 All characters can have spells from racial abilities (e.g., Tiefling), feats (Magic Initiate), or magic items.")
        
        # Initialize defaults from existing spellbook or empty
        has_spellbook = character.spellbook is not None
        
        new_spell_dc = character.spellbook.spell_save_dc if has_spellbook else 10
        new_spell_attack = character.spellbook.spell_attack_bonus if has_spellbook else 0
        new_cantrips_str = ", ".join(character.spellbook.cantrips) if has_spellbook else ""
        new_spells_str = ", ".join(character.spellbook.spells_known) if has_spellbook else ""
        new_spell_slots = {}
        
        st.markdown("#### Spellcasting Ability")
        col1, col2 = st.columns(2)
        new_spell_dc = col1.number_input("Spell Save DC", 1, 30, new_spell_dc, key=f"edit_spell_dc_{uid}")
        new_spell_attack = col2.number_input("Spell Attack Bonus", -5, 20, new_spell_attack, key=f"edit_spell_atk_{uid}")
        
        st.divider()
        
        st.markdown("#### Spell Slots")
        st.caption("Set max slots per level. Leave at 0 for non-casters.")
        slot_cols = st.columns(9)
        for level in range(1, 10):
            current_slots = (0, 0)
            if has_spellbook:
                current_slots = character.spellbook.spell_slots.get(level, (0, 0))
            max_val = slot_cols[level-1].number_input(
                f"L{level}", 0, 10, current_slots[1], key=f"edit_slot_max_{level}_{uid}"
            )
            if max_val > 0:
                new_spell_slots[level] = (max_val, max_val)
        
        st.divider()
        
        st.markdown("#### Cantrips")
        st.caption("Include racial cantrips (e.g., Thaumaturgy for Tieflings)")
        new_cantrips_str = st.text_area("Cantrips (comma-separated)", value=new_cantrips_str, key=f"edit_cantrips_{uid}", height=80)
        
        st.markdown("#### Spells Known/Prepared")
        st.caption("Include racial spells (e.g., Hellish Rebuke for Tieflings at level 3)")
        new_spells_str = st.text_area("Spells (comma-separated)", value=new_spells_str, key=f"edit_spells_{uid}", height=120)
    
    # =========================================================================
    # TAB 5: Equipment
    # =========================================================================
    with tab_equipment:
        st.markdown("#### Currency")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        current_cp = character.inventory.currency_cp if character.inventory else 0
        current_gp = current_cp // 100
        current_sp = (current_cp % 100) // 10
        current_cp_only = current_cp % 10
        
        new_gp = col1.number_input("Gold (gp)", 0, 99999, current_gp, key=f"edit_gp_{uid}")
        new_sp = col2.number_input("Silver (sp)", 0, 99, (current_cp % 100) // 10, key=f"edit_sp_{uid}")
        new_cp = col3.number_input("Copper (cp)", 0, 99, current_cp % 10, key=f"edit_cp_{uid}")
        new_ep = col4.number_input("Electrum (ep)", 0, 999, 0, key=f"edit_ep_{uid}")
        new_pp = col5.number_input("Platinum (pp)", 0, 999, 0, key=f"edit_pp_{uid}")
        
        st.divider()
        
        st.markdown("#### Equipment & Items")
        st.caption("List your equipment, one item per line. Format: `Item Name` or `Item Name (x2)` for quantity")
        
        # Build current inventory string
        inv_lines = []
        if character.inventory:
            for item in character.inventory.items:
                qty_str = f" (x{item.quantity})" if item.quantity > 1 else ""
                equipped_str = " [EQUIPPED]" if item.equipped else ""
                inv_lines.append(f"{item.name}{qty_str}{equipped_str}")
        
        new_inventory_str = st.text_area(
            "Equipment List",
            value="\n".join(inv_lines),
            height=200,
            key=f"edit_inventory_{uid}",
            help="One item per line. Add [EQUIPPED] to mark as equipped."
        )
        
        st.divider()
        
        st.markdown("#### Features & Traits")
        features_str = "\n".join(character.class_features.features) if character.class_features else ""
        new_features_str = st.text_area(
            "Features (one per line)",
            value=features_str,
            height=150,
            key=f"edit_features_{uid}"
        )
    
    # =========================================================================
    # TAB 6: Personality
    # =========================================================================
    with tab_personality:
        journal = character.journal
        
        st.markdown("#### Personality Traits")
        traits_str = "\n".join(journal.personality_traits) if journal else ""
        new_traits_str = st.text_area("Personality Traits (one per line)", value=traits_str, height=80, key=f"edit_traits_{uid}")
        
        st.markdown("#### Ideals")
        ideals_str = "\n".join(journal.ideals) if journal else ""
        new_ideals_str = st.text_area("Ideals (one per line)", value=ideals_str, height=80, key=f"edit_ideals_{uid}")
        
        st.markdown("#### Bonds")
        bonds_str = "\n".join(journal.bonds) if journal else ""
        new_bonds_str = st.text_area("Bonds (one per line)", value=bonds_str, height=80, key=f"edit_bonds_{uid}")
        
        st.markdown("#### Flaws")
        flaws_str = "\n".join(journal.flaws) if journal else ""
        new_flaws_str = st.text_area("Flaws (one per line)", value=flaws_str, height=80, key=f"edit_flaws_{uid}")
        
        st.divider()
        
        st.markdown("#### Backstory")
        backstory = journal.backstory if journal else ""
        new_backstory = st.text_area("Character Backstory", value=backstory, height=200, key=f"edit_backstory_{uid}")
        
        st.markdown("#### Allies & Organizations")
        allies = journal.allies_and_organizations if journal else ""
        new_allies = st.text_area("Allies & Organizations", value=allies, height=100, key=f"edit_allies_{uid}")
    
    # =========================================================================
    # TAB 7: Appearance
    # =========================================================================
    with tab_appearance:
        journal = character.journal
        
        st.markdown("#### Physical Characteristics")
        col1, col2, col3 = st.columns(3)
        new_age = col1.text_input("Age", value=journal.age if journal else "", key=f"edit_age_{uid}")
        new_height = col2.text_input("Height", value=journal.height if journal else "", key=f"edit_height_{uid}")
        new_weight = col3.text_input("Weight", value=journal.weight if journal else "", key=f"edit_weight_{uid}")
        
        col1, col2, col3 = st.columns(3)
        new_eyes = col1.text_input("Eyes", value=journal.eyes if journal else "", key=f"edit_eyes_{uid}")
        new_hair = col2.text_input("Hair", value=journal.hair if journal else "", key=f"edit_hair_{uid}")
        new_skin = col3.text_input("Skin", value=journal.skin if journal else "", key=f"edit_skin_{uid}")
        
        st.divider()
        
        st.markdown("#### Appearance Description")
        appearance = journal.appearance if journal else ""
        new_appearance = st.text_area("Physical Appearance", value=appearance, height=150, key=f"edit_appearance_{uid}")
        
        st.divider()
        
        st.markdown("#### Additional Treasure")
        treasure = journal.treasure if journal else ""
        new_treasure = st.text_area("Treasure & Valuables", value=treasure, height=100, key=f"edit_treasure_{uid}")
        
        st.markdown("#### Additional Features/Traits")
        add_features = journal.additional_features_traits if journal else ""
        new_add_features = st.text_area("Additional Notes", value=add_features, height=100, key=f"edit_add_features_{uid}")
    
    # =========================================================================
    # Save/Cancel Buttons (always visible)
    # =========================================================================
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("💾 Save All Changes", key=f"save_char_{uid}", type="primary"):
            # Apply all changes
            
            # Basic Info
            character.name = new_name
            character.race = new_race
            character.alignment = new_alignment
            character.size = new_size
            character.background = new_background
            
            # Stats
            character.stats.strength = new_str
            character.stats.dexterity = new_dex
            character.stats.constitution = new_con
            character.stats.intelligence = new_int
            character.stats.wisdom = new_wis
            character.stats.charisma = new_cha
            character.stats.proficiency_bonus = new_prof_bonus
            character.stats.save_proficiencies = new_save_profs
            character.stats.skill_proficiencies = new_skill_profs
            
            # Health
            character.health.hp_current = new_hp_current
            character.health.hp_max = new_hp_max
            character.health.hp_temp = new_hp_temp
            character.health.death_saves_success = new_death_success
            character.health.death_saves_failure = new_death_fail
            
            # Defense
            character.defense.ac_base = new_ac_base
            character.defense.ac_armor = new_ac_armor
            character.defense.ac_shield = new_ac_shield
            
            # Movement
            if character.movement:
                character.movement.speed_walk = new_speed
                character.movement.speed_fly = new_fly
                character.movement.speed_swim = new_swim
                character.movement.speed_climb = new_climb
                character.movement.movement_remaining = new_speed
            
            # Class
            if character.class_features and character.class_features.classes:
                character.class_features.classes[0].class_name = new_class
                character.class_features.classes[0].level = new_level
                character.class_features.classes[0].subclass = new_subclass if new_subclass else None
            
            character.experience_points = new_xp
            
            # Features
            if character.class_features:
                character.class_features.features = [f.strip() for f in new_features_str.split("\n") if f.strip()]
            
            # Spellcasting - create spellbook if user added cantrips/spells
            new_cantrips = [c.strip() for c in new_cantrips_str.split(",") if c.strip()]
            new_spells = [s.strip() for s in new_spells_str.split(",") if s.strip()]
            has_spell_data = new_cantrips or new_spells or any(v[1] > 0 for v in new_spell_slots.values())
            
            if character.spellbook:
                character.spellbook.spell_save_dc = new_spell_dc
                character.spellbook.spell_attack_bonus = new_spell_attack
                character.spellbook.cantrips = new_cantrips
                character.spellbook.spells_known = new_spells
                character.spellbook.spell_slots = new_spell_slots
            elif has_spell_data:
                # Create spellbook for racial/feat spells
                from dnd_manager.models.ecs import SpellbookComponent
                character.spellbook = SpellbookComponent(
                    spell_save_dc=new_spell_dc,
                    spell_attack_bonus=new_spell_attack,
                    cantrips=new_cantrips,
                    spells_known=new_spells,
                    spell_slots=new_spell_slots,
                )
            
            # Inventory
            if character.inventory:
                # Parse inventory text
                new_items = []
                for line in new_inventory_str.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    
                    equipped = "[EQUIPPED]" in line.upper()
                    line = line.replace("[EQUIPPED]", "").replace("[equipped]", "").strip()
                    
                    # Parse quantity
                    quantity = 1
                    import re
                    qty_match = re.search(r'\(x?(\d+)\)$', line)
                    if qty_match:
                        quantity = int(qty_match.group(1))
                        line = line[:qty_match.start()].strip()
                    
                    if line:
                        new_items.append(ItemStack(
                            name=line,
                            quantity=quantity,
                            equipped=equipped,
                            item_id=line.lower().replace(" ", "_"),
                        ))
                
                character.inventory.items = new_items
                character.inventory.currency_cp = (new_gp * 100) + (new_sp * 10) + new_cp + (new_ep * 50) + (new_pp * 1000)
            
            # Journal/Personality
            if character.journal:
                character.journal.personality_traits = [t.strip() for t in new_traits_str.split("\n") if t.strip()]
                character.journal.ideals = [i.strip() for i in new_ideals_str.split("\n") if i.strip()]
                character.journal.bonds = [b.strip() for b in new_bonds_str.split("\n") if b.strip()]
                character.journal.flaws = [f.strip() for f in new_flaws_str.split("\n") if f.strip()]
                character.journal.backstory = new_backstory
                character.journal.allies_and_organizations = new_allies
                character.journal.age = new_age
                character.journal.height = new_height
                character.journal.weight = new_weight
                character.journal.eyes = new_eyes
                character.journal.hair = new_hair
                character.journal.skin = new_skin
                character.journal.appearance = new_appearance
                character.journal.treasure = new_treasure
                character.journal.additional_features_traits = new_add_features
            
            # Save and close editor
            save_characters()
            st.session_state[f"editing_{uid}"] = False
            st.toast(f"Saved changes to {character.name}!", icon="✅")
            st.rerun()
    
    with col2:
        if st.button("❌ Cancel", key=f"cancel_edit_{uid}"):
            st.session_state[f"editing_{uid}"] = False
            st.rerun()


def render_character_details(character: ActorEntity) -> None:
    """Render detailed character information."""
    
    # Ability Scores
    st.markdown("**Ability Scores**")
    cols = st.columns(6)
    abilities = [
        ("STR", character.stats.strength),
        ("DEX", character.stats.dexterity),
        ("CON", character.stats.constitution),
        ("INT", character.stats.intelligence),
        ("WIS", character.stats.wisdom),
        ("CHA", character.stats.charisma),
    ]
    for i, (name, score) in enumerate(abilities):
        mod = (score - 10) // 2
        mod_str = f"+{mod}" if mod >= 0 else str(mod)
        cols[i].metric(name, score, mod_str)
    
    st.divider()
    
    # Combat Stats
    st.markdown("**Combat**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AC", character.ac)
    col2.metric("HP", f"{character.health.hp_current}/{character.health.hp_max}")
    col3.metric("Speed", f"{character.movement.speed_walk if character.movement else 30} ft")
    col4.metric("Prof. Bonus", f"+{character.stats.proficiency_bonus}")
    
    st.divider()
    
    # Skills
    st.markdown("**Skills**")
    
    from dnd_manager.models.progression import SKILL_ABILITIES, ALL_SKILLS
    
    # Build skill display - 3 columns
    skill_cols = st.columns(3)
    
    stats = character.stats
    prof_bonus = stats.proficiency_bonus
    
    for i, skill in enumerate(sorted(ALL_SKILLS)):
        ability = SKILL_ABILITIES.get(skill, "dexterity")
        ability_mod = getattr(stats, f"{ability[:3]}_mod", 0)
        
        # Check proficiency (1 = proficient, 2 = expertise)
        prof_mult = stats.skill_proficiencies.get(skill, 0)
        
        total = ability_mod + (prof_bonus * prof_mult)
        
        # Format display
        skill_name = skill.replace("_", " ").title()
        
        if prof_mult == 2:
            display = f"★★ **{skill_name}**: +{total}"
        elif prof_mult == 1:
            display = f"★ **{skill_name}**: +{total}"
        else:
            sign = "+" if total >= 0 else ""
            display = f"{skill_name}: {sign}{total}"
        
        col_idx = i % 3
        skill_cols[col_idx].markdown(display, unsafe_allow_html=True)
    
    st.divider()
    
    # Saving Throws
    st.markdown("**Saving Throws**")
    save_cols = st.columns(6)
    
    for i, ability in enumerate(["strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"]):
        ability_mod = getattr(stats, f"{ability[:3]}_mod", 0)
        
        # Check if proficient (handle both enum and string)
        from dnd_manager.models.ecs import Ability
        save_prof_values = []
        for s in stats.save_proficiencies:
            if hasattr(s, 'value'):
                save_prof_values.append(s.value.lower())
            else:
                save_prof_values.append(str(s).lower())
        is_prof = ability.lower() in save_prof_values
        
        total = ability_mod + (prof_bonus if is_prof else 0)
        
        label = ability[:3].upper()
        sign = "+" if total >= 0 else ""
        
        if is_prof:
            save_cols[i].markdown(f"★ **{label}**: {sign}{total}")
        else:
            save_cols[i].markdown(f"{label}: {sign}{total}")
    
    st.divider()
    
    # Equipment
    if character.inventory and character.inventory.items:
        st.markdown("**Equipment**")
        
        equipped = [item for item in character.inventory.items if item.equipped]
        other = [item for item in character.inventory.items if not item.equipped]
        
        if equipped:
            st.markdown("*Equipped:*")
            for item in equipped:
                details = []
                if item.damage_dice:
                    details.append(f"{item.damage_dice} {item.damage_type or ''}")
                if item.ac_base:
                    details.append(f"AC {item.ac_base}")
                if item.ac_bonus:
                    details.append(f"+{item.ac_bonus} AC")
                detail_str = f" ({', '.join(details)})" if details else ""
                st.markdown(f"- ⚔️ **{item.name}**{detail_str}")
        
        if other:
            st.markdown("*Inventory:*")
            # Group by type and show counts
            item_counts = {}
            for item in other:
                key = item.name
                if key in item_counts:
                    item_counts[key] += item.quantity
                else:
                    item_counts[key] = item.quantity
            
            for name, qty in list(item_counts.items())[:15]:  # Limit display
                qty_str = f" (x{qty})" if qty > 1 else ""
                st.markdown(f"- {name}{qty_str}")
            
            if len(item_counts) > 15:
                st.caption(f"... and {len(item_counts) - 15} more items")
        
        # Gold
        if character.inventory.currency_cp > 0:
            gp = character.inventory.currency_cp // 100
            sp = (character.inventory.currency_cp % 100) // 10
            cp = character.inventory.currency_cp % 10
            currency_parts = []
            if gp: currency_parts.append(f"{gp} gp")
            if sp: currency_parts.append(f"{sp} sp")
            if cp: currency_parts.append(f"{cp} cp")
            st.markdown(f"💰 **Currency:** {', '.join(currency_parts) or '0 gp'}")
    
    st.divider()
    
    # Spells (if any)
    if character.spellbook:
        st.markdown("**Spellcasting**")
        
        if character.spellbook.cantrips:
            st.markdown(f"*Cantrips:* {', '.join(character.spellbook.cantrips)}")
        
        if character.spellbook.spells_known:
            st.markdown(f"*Spells Known:* {', '.join(character.spellbook.spells_known)}")
        
        if character.spellbook.spell_slots:
            slots_str = ", ".join(
                f"L{level}: {current}/{max_val}" 
                for level, (current, max_val) in sorted(character.spellbook.spell_slots.items())
            )
            st.markdown(f"*Spell Slots:* {slots_str}")
        
        st.divider()
    
    # Features & Traits
    if character.class_features and character.class_features.features:
        st.markdown("**Features & Traits**")
        for feature in character.class_features.features[:10]:
            st.markdown(f"- {feature}")
        if len(character.class_features.features) > 10:
            st.caption(f"... and {len(character.class_features.features) - 10} more")
        st.divider()
    
    # Background
    if character.background:
        st.markdown(f"**Background:** {character.background}")
        st.divider()
    
    # Personality (from Journal)
    if character.journal:
        has_personality = (
            character.journal.personality_traits or 
            character.journal.ideals or 
            character.journal.bonds or 
            character.journal.flaws
        )
        if has_personality:
            st.markdown("**Personality**")
            if character.journal.personality_traits:
                st.markdown(f"*Traits:* {', '.join(character.journal.personality_traits)}")
            if character.journal.ideals:
                st.markdown(f"*Ideals:* {', '.join(character.journal.ideals)}")
            if character.journal.bonds:
                st.markdown(f"*Bonds:* {', '.join(character.journal.bonds)}")
            if character.journal.flaws:
                st.markdown(f"*Flaws:* {', '.join(character.journal.flaws)}")
            st.divider()
        
        # Appearance
        appearance_summary = character.journal.get_appearance_summary() if hasattr(character.journal, 'get_appearance_summary') else ""
        if not appearance_summary:
            # Fallback: build from individual fields
            appearance_parts = []
            if character.journal.age:
                appearance_parts.append(f"Age: {character.journal.age}")
            if character.journal.height:
                appearance_parts.append(f"Height: {character.journal.height}")
            if character.journal.weight:
                appearance_parts.append(f"Weight: {character.journal.weight}")
            if character.journal.eyes:
                appearance_parts.append(f"Eyes: {character.journal.eyes}")
            if character.journal.hair:
                appearance_parts.append(f"Hair: {character.journal.hair}")
            if character.journal.skin:
                appearance_parts.append(f"Skin: {character.journal.skin}")
            if character.journal.appearance:
                appearance_parts.append(character.journal.appearance)
            appearance_summary = "; ".join(appearance_parts)
        
        if appearance_summary:
            st.markdown("**Appearance**")
            st.markdown(appearance_summary)
            st.divider()
        
        # Backstory
        if character.journal.backstory:
            st.markdown("**Backstory**")
            st.markdown(character.journal.backstory)
            st.divider()
        
        # Allies & Organizations
        if character.journal.allies_and_organizations:
            st.markdown("**Allies & Organizations**")
            st.markdown(character.journal.allies_and_organizations)
            st.divider()
        
        # Additional Treasure
        if character.journal.treasure:
            st.markdown("**Additional Treasure**")
            st.markdown(character.journal.treasure)
            st.divider()
        
        # Additional Features & Traits
        if character.journal.additional_features_traits:
            st.markdown("**Additional Features & Traits**")
            st.markdown(character.journal.additional_features_traits)


def roll_4d6_drop_lowest() -> int:
    """Roll 4d6 and drop the lowest die."""
    import random
    rolls = [random.randint(1, 6) for _ in range(4)]
    rolls.sort(reverse=True)
    return sum(rolls[:3])  # Sum top 3


def render_character_creator() -> None:
    """Render the character creation form."""
    
    st.markdown("### Create New Character")
    st.markdown("*Follow D&D 5e character creation rules*")
    
    # Initialize session state for rolled stats
    if "cc_rolled_stats" not in st.session_state:
        st.session_state.cc_rolled_stats = None
    
    # Basic info
    st.markdown("#### Basic Information")
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Character Name", key="cc_name")
        race = st.selectbox("Race", RACES, key="cc_race")
        background = st.selectbox("Background", BACKGROUNDS, key="cc_background")
    
    with col2:
        char_class = st.selectbox("Class", CLASSES, key="cc_class")
        level = st.number_input("Level", min_value=1, max_value=20, value=1, key="cc_level")
        alignment = st.selectbox("Alignment", ALIGNMENTS, key="cc_alignment")
    
    st.divider()
    
    # Ability Scores with method selection
    st.markdown("#### Ability Scores")
    
    stat_method = st.radio(
        "Generation Method",
        options=list(STAT_METHODS.keys()),
        format_func=lambda x: STAT_METHODS[x],
        key="cc_stat_method",
        horizontal=True,
    )
    
    stats_valid = False
    strength = dexterity = constitution = intelligence = wisdom = charisma = 10
    
    if stat_method == "standard_array":
        st.markdown("*Assign each value from the standard array exactly once: 15, 14, 13, 12, 10, 8*")
        
        # Initialize session state for standard array assignments
        if "sa_assignments" not in st.session_state:
            st.session_state.sa_assignments = {
                "strength": None,
                "dexterity": None,
                "constitution": None,
                "intelligence": None,
                "wisdom": None,
                "charisma": None,
            }
        
        # Helper function to get available options for an ability
        def get_available_options(current_ability: str) -> list[int]:
            """Get available values excluding those already assigned to other abilities."""
            assigned_values = [
                val for ability, val in st.session_state.sa_assignments.items()
                if ability != current_ability and val is not None
            ]
            current_value = st.session_state.sa_assignments[current_ability]
            
            # Available values are those not assigned elsewhere
            available = [v for v in STANDARD_ARRAY if v not in assigned_values]
            
            # If current ability has a value, ensure it's in the list
            if current_value is not None and current_value not in available:
                available.insert(0, current_value)
            
            # Sort for better UX
            return sorted(available, reverse=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strength_options = get_available_options("strength")
            strength_idx = 0
            if st.session_state.sa_assignments["strength"] in strength_options:
                strength_idx = strength_options.index(st.session_state.sa_assignments["strength"])
            strength = st.selectbox("Strength", strength_options, index=strength_idx, key="cc_str_sa")
            st.session_state.sa_assignments["strength"] = strength
            
            intelligence_options = get_available_options("intelligence")
            intelligence_idx = 0
            if st.session_state.sa_assignments["intelligence"] in intelligence_options:
                intelligence_idx = intelligence_options.index(st.session_state.sa_assignments["intelligence"])
            intelligence = st.selectbox("Intelligence", intelligence_options, index=intelligence_idx, key="cc_int_sa")
            st.session_state.sa_assignments["intelligence"] = intelligence
        
        with col2:
            dexterity_options = get_available_options("dexterity")
            dexterity_idx = 0
            if st.session_state.sa_assignments["dexterity"] in dexterity_options:
                dexterity_idx = dexterity_options.index(st.session_state.sa_assignments["dexterity"])
            dexterity = st.selectbox("Dexterity", dexterity_options, index=dexterity_idx, key="cc_dex_sa")
            st.session_state.sa_assignments["dexterity"] = dexterity
            
            wisdom_options = get_available_options("wisdom")
            wisdom_idx = 0
            if st.session_state.sa_assignments["wisdom"] in wisdom_options:
                wisdom_idx = wisdom_options.index(st.session_state.sa_assignments["wisdom"])
            wisdom = st.selectbox("Wisdom", wisdom_options, index=wisdom_idx, key="cc_wis_sa")
            st.session_state.sa_assignments["wisdom"] = wisdom
        
        with col3:
            constitution_options = get_available_options("constitution")
            constitution_idx = 0
            if st.session_state.sa_assignments["constitution"] in constitution_options:
                constitution_idx = constitution_options.index(st.session_state.sa_assignments["constitution"])
            constitution = st.selectbox("Constitution", constitution_options, index=constitution_idx, key="cc_con_sa")
            st.session_state.sa_assignments["constitution"] = constitution
            
            charisma_options = get_available_options("charisma")
            charisma_idx = 0
            if st.session_state.sa_assignments["charisma"] in charisma_options:
                charisma_idx = charisma_options.index(st.session_state.sa_assignments["charisma"])
            charisma = st.selectbox("Charisma", charisma_options, index=charisma_idx, key="cc_cha_sa")
            st.session_state.sa_assignments["charisma"] = charisma
        
        # Validate that all values are assigned
        assigned = [strength, dexterity, constitution, intelligence, wisdom, charisma]
        if None not in assigned and sorted(assigned) == sorted(STANDARD_ARRAY):
            stats_valid = True
            st.success("✓ Valid standard array assignment")
        elif len(set(assigned)) < len(assigned):
            st.warning("⚠️ Some values are used more than once")
        else:
            remaining = len([v for v in assigned if v is None])
            if remaining > 0:
                st.info(f"Assign all ability scores ({remaining} remaining)")
    
    elif stat_method == "point_buy":
        st.markdown(f"*Spend {POINT_BUY_TOTAL} points. Scores range from 8-15.*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strength = st.slider("Strength", 8, 15, 10, key="cc_str_pb")
            intelligence = st.slider("Intelligence", 8, 15, 10, key="cc_int_pb")
        
        with col2:
            dexterity = st.slider("Dexterity", 8, 15, 10, key="cc_dex_pb")
            wisdom = st.slider("Wisdom", 8, 15, 10, key="cc_wis_pb")
        
        with col3:
            constitution = st.slider("Constitution", 8, 15, 10, key="cc_con_pb")
            charisma = st.slider("Charisma", 8, 15, 10, key="cc_cha_pb")
        
        # Calculate point cost
        total_cost = sum([
            POINT_BUY_COSTS[strength], POINT_BUY_COSTS[dexterity],
            POINT_BUY_COSTS[constitution], POINT_BUY_COSTS[intelligence],
            POINT_BUY_COSTS[wisdom], POINT_BUY_COSTS[charisma]
        ])
        
        remaining = POINT_BUY_TOTAL - total_cost
        
        if remaining == 0:
            st.success(f"✓ Points spent: {total_cost}/{POINT_BUY_TOTAL}")
            stats_valid = True
        elif remaining > 0:
            st.warning(f"Points spent: {total_cost}/{POINT_BUY_TOTAL} ({remaining} remaining)")
        else:
            st.error(f"Over budget! Points spent: {total_cost}/{POINT_BUY_TOTAL} ({-remaining} over)")
    
    elif stat_method == "roll":
        st.markdown("*Roll 4d6, drop the lowest die for each ability score*")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("🎲 Roll Stats", key="roll_stats_btn", use_container_width=True):
                st.session_state.cc_rolled_stats = [roll_4d6_drop_lowest() for _ in range(6)]
                st.rerun()
        
        if st.session_state.cc_rolled_stats:
            rolled = st.session_state.cc_rolled_stats
            st.markdown(f"**Rolled:** {', '.join(str(r) for r in sorted(rolled, reverse=True))}")
            
            st.markdown("*Assign each rolled value to an ability:*")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                strength = st.selectbox("Strength", sorted(rolled, reverse=True), key="cc_str_roll")
                intelligence = st.selectbox("Intelligence", sorted(rolled, reverse=True), key="cc_int_roll")
            
            with col2:
                dexterity = st.selectbox("Dexterity", sorted(rolled, reverse=True), key="cc_dex_roll")
                wisdom = st.selectbox("Wisdom", sorted(rolled, reverse=True), key="cc_wis_roll")
            
            with col3:
                constitution = st.selectbox("Constitution", sorted(rolled, reverse=True), key="cc_con_roll")
                charisma = st.selectbox("Charisma", sorted(rolled, reverse=True), key="cc_cha_roll")
            
            assigned = [strength, dexterity, constitution, intelligence, wisdom, charisma]
            if sorted(assigned) == sorted(rolled):
                stats_valid = True
                st.success("✓ Valid assignment")
            else:
                st.error("Each rolled value must be used exactly once!")
        else:
            st.info("Click 'Roll Stats' to generate your ability scores")
            strength = dexterity = constitution = intelligence = wisdom = charisma = 10
    
    else:  # manual
        st.markdown("*Enter your stats manually (3-18 each)*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strength = st.number_input("Strength", 3, 18, 10, key="cc_str_man")
            intelligence = st.number_input("Intelligence", 3, 18, 10, key="cc_int_man")
        
        with col2:
            dexterity = st.number_input("Dexterity", 3, 18, 10, key="cc_dex_man")
            wisdom = st.number_input("Wisdom", 3, 18, 10, key="cc_wis_man")
        
        with col3:
            constitution = st.number_input("Constitution", 3, 18, 10, key="cc_con_man")
            charisma = st.number_input("Charisma", 3, 18, 10, key="cc_cha_man")
        
        stats_valid = True  # Manual entry is always valid
    
    st.divider()
    
    # Character Details (Appearance, Backstory, etc.)
    st.markdown("#### Character Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.text_input("Age", placeholder="e.g., 25", key="cc_age")
        height = st.text_input("Height", placeholder="e.g., 5'10\"", key="cc_height")
        weight = st.text_input("Weight", placeholder="e.g., 170 lbs", key="cc_weight")
        eyes = st.text_input("Eyes", placeholder="e.g., Blue", key="cc_eyes")
    
    with col2:
        skin = st.text_input("Skin", placeholder="e.g., Fair", key="cc_skin")
        hair = st.text_input("Hair", placeholder="e.g., Brown, long", key="cc_hair")
        gender = st.text_input("Gender", placeholder="e.g., Male, Female, Non-binary", key="cc_gender")
    
    appearance = st.text_area(
        "Appearance",
        placeholder="Describe your character's physical appearance, distinguishing features, clothing style...",
        key="cc_appearance",
        height=80,
    )
    
    st.divider()
    
    # Skill Proficiencies
    st.markdown("#### Skill Proficiencies")
    
    from dnd_manager.models.progression import (
        get_skill_options,
        get_skill_count,
        SKILL_ABILITIES,
        get_racial_traits,
        get_all_features_up_to_level,
    )
    
    skill_options = get_skill_options(char_class)
    skill_count = get_skill_count(char_class)
    
    # Get racial skill proficiencies
    racial_traits = get_racial_traits(race)
    racial_skills = racial_traits.get("skill_proficiencies", [])
    extra_racial_skills = racial_traits.get("extra_skill_count", 0)
    
    # Display what skills are fixed from race
    if racial_skills:
        st.caption(f"*From {race}: {', '.join(s.replace('_', ' ').title() for s in racial_skills)}*")
    
    st.markdown(f"Choose **{skill_count}** skills from your class options:")
    
    # Format skills for display
    skill_display = []
    for skill in skill_options:
        ability = SKILL_ABILITIES.get(skill, "?")
        display = f"{skill.replace('_', ' ').title()} ({ability[:3].upper()})"
        skill_display.append((skill, display))
    
    # Multi-select for skills
    selected_skill_displays = st.multiselect(
        "Select Skills",
        [d for _, d in skill_display],
        max_selections=skill_count,
        key="cc_skills",
        label_visibility="collapsed",
    )
    
    # Map back to skill names
    display_to_skill = {d: s for s, d in skill_display}
    selected_skills = [display_to_skill.get(d, "") for d in selected_skill_displays]
    
    # Validate skill selection
    skills_valid = len(selected_skills) == skill_count
    if skills_valid:
        st.success(f"✓ Selected {skill_count} skills")
    else:
        st.warning(f"Select {skill_count - len(selected_skills)} more skill(s)")
    
    # Extra skills from Half-Elf
    extra_skills = []
    if extra_racial_skills > 0:
        st.markdown(f"*{race} Bonus: Choose {extra_racial_skills} additional skill(s) from any:*")
        all_skill_display = [(s, f"{s.replace('_', ' ').title()} ({SKILL_ABILITIES.get(s, '?')[:3].upper()})") 
                            for s in SKILL_ABILITIES.keys() if s not in selected_skills and s not in racial_skills]
        
        extra_skill_displays = st.multiselect(
            "Bonus Skills",
            [d for _, d in all_skill_display],
            max_selections=extra_racial_skills,
            key="cc_extra_skills",
            label_visibility="collapsed",
        )
        
        extra_display_to_skill = {d: s for s, d in all_skill_display}
        extra_skills = [extra_display_to_skill.get(d, "") for d in extra_skill_displays]
    
    st.divider()
    
    st.markdown("#### Personality & Background")
    
    personality_traits = st.text_area(
        "Personality Traits",
        placeholder="Two personality traits that define your character...",
        key="cc_personality",
        height=60,
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        ideals = st.text_area(
            "Ideals",
            placeholder="What principles guide your character?",
            key="cc_ideals",
            height=60,
        )
        bonds = st.text_area(
            "Bonds",
            placeholder="What ties your character to people, places, or events?",
            key="cc_bonds",
            height=60,
        )
    
    with col2:
        flaws = st.text_area(
            "Flaws",
            placeholder="What weaknesses or vices does your character have?",
            key="cc_flaws",
            height=60,
        )
        backstory = st.text_area(
            "Backstory",
            placeholder="Your character's history and background...",
            key="cc_backstory",
            height=60,
        )
    
    st.divider()
    
    # Calculate derived stats
    con_mod = (constitution - 10) // 2
    dex_mod = (dexterity - 10) // 2
    hit_die = CLASS_HIT_DIE.get(char_class, 8)
    prof_bonus = 2 + (level - 1) // 4
    
    # HP calculation
    if level == 1:
        hp = hit_die + con_mod
    else:
        avg_roll = (hit_die // 2) + 1
        hp = hit_die + con_mod + (avg_roll + con_mod) * (level - 1)
    hp = max(hp, 1)
    
    # Display calculated stats
    st.markdown("#### Calculated Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hit Points", hp)
    col2.metric("AC (Unarmored)", 10 + dex_mod)
    col3.metric("Proficiency", f"+{prof_bonus}")
    col4.metric("Hit Die", f"d{hit_die}")
    
    # Show saving throw proficiencies
    save_profs = CLASS_SAVE_PROFS.get(char_class, [])
    st.caption(f"**Saving Throw Proficiencies:** {', '.join(s.title() for s in save_profs)}")
    
    st.divider()
    
    # Create button
    can_create = stats_valid and skills_valid and name.strip()
    
    if st.button(
        "✨ Create Character", 
        key="create_char_btn", 
        use_container_width=True,
        disabled=not can_create,
    ):
        if not name.strip():
            st.error("Please enter a character name")
            return
        
        if not stats_valid:
            st.error("Please fix ability score assignment")
            return
        
        if not skills_valid:
            st.error(f"Please select {skill_count} skills")
            return
        
        # Half-Elf Ability Score Choice
        half_elf_bonuses = {}
        if "Half-Elf" in race:
            st.markdown("#### Half-Elf Bonus Abilities")
            st.markdown("*Choose two abilities to increase by +1 (in addition to +2 Charisma):*")
            
            # Available abilities (all except Charisma which already gets +2)
            available_abilities = ["Strength", "Dexterity", "Constitution", "Intelligence", "Wisdom"]
            
            chosen_abilities = st.multiselect(
                "Select two abilities for +1 bonus",
                available_abilities,
                max_selections=2,
                key="half_elf_bonus_abilities",
                help="Half-Elves get +2 to Charisma and +1 to two other abilities of your choice"
            )
            
            if len(chosen_abilities) == 2:
                st.success(f"✓ Half-Elf bonuses: +2 Charisma, +1 {chosen_abilities[0]}, +1 {chosen_abilities[1]}")
                # Convert to lowercase keys for application
                for ability in chosen_abilities:
                    half_elf_bonuses[ability.lower()] = 1
            elif len(chosen_abilities) < 2:
                st.warning(f"⚠️ Please select 2 abilities ({2 - len(chosen_abilities)} remaining)")
            
            # Prevent finishing character creation without making the choice
            if len(chosen_abilities) != 2:
                st.error("Half-Elves must choose exactly 2 abilities for +1 bonus")
                return
        
        # Apply racial ability bonuses
        ability_bonuses = racial_traits.get("ability_bonuses", {})
        # Apply Half-Elf chosen bonuses
        for ability, bonus in half_elf_bonuses.items():
            ability_bonuses[ability] = ability_bonuses.get(ability, 0) + bonus
        strength += ability_bonuses.get("strength", 0)
        dexterity += ability_bonuses.get("dexterity", 0)
        constitution += ability_bonuses.get("constitution", 0)
        intelligence += ability_bonuses.get("intelligence", 0)
        wisdom += ability_bonuses.get("wisdom", 0)
        charisma += ability_bonuses.get("charisma", 0)
        
        # Cap at 20
        strength = min(20, strength)
        dexterity = min(20, dexterity)
        constitution = min(20, constitution)
        intelligence = min(20, intelligence)
        wisdom = min(20, wisdom)
        charisma = min(20, charisma)
        
        # Recalculate derived stats with bonuses
        con_mod = (constitution - 10) // 2
        dex_mod = (dexterity - 10) // 2
        
        # HP with racial bonus (Hill Dwarf)
        hp_bonus = racial_traits.get("hp_bonus_per_level", 0) * level
        if level == 1:
            hp = hit_die + con_mod + hp_bonus
        else:
            avg_roll = (hit_die // 2) + 1
            hp = hit_die + con_mod + (avg_roll + con_mod) * (level - 1) + hp_bonus
        hp = max(hp, 1)
        
        # Build save proficiencies
        save_prof_enums = []
        for save in save_profs:
            try:
                save_prof_enums.append(Ability(save))
            except ValueError:
                pass
        
        # Build skill proficiencies dict (skill -> 1 for proficient)
        all_skill_profs = selected_skills + racial_skills + extra_skills
        skill_prof_dict = {skill: 1 for skill in all_skill_profs if skill}
        
        # Get class features for starting level
        class_features_list = get_all_features_up_to_level(char_class, level)
        
        # Add racial traits as features
        racial_trait_names = racial_traits.get("traits", [])
        all_features = racial_trait_names + class_features_list
        
        # Build tracked features with proper usage limits
        from dnd_manager.models.progression import get_feature_resource_info
        from dnd_manager.models.ecs import Feature, RechargeType
        
        tracked_features: dict[str, Feature] = {}
        simple_features: list[str] = []
        
        stats_for_features = {
            "strength": strength,
            "dexterity": dexterity,
            "constitution": constitution,
            "intelligence": intelligence,
            "wisdom": wisdom,
            "charisma": charisma,
        }
        
        for feature_name in all_features:
            resource_info = get_feature_resource_info(feature_name, level, stats_for_features)
            if resource_info and resource_info.get("uses_max") is not None:
                # This feature has limited uses - track it
                recharge_map = {
                    "short_rest": RechargeType.SHORT_REST,
                    "long_rest": RechargeType.LONG_REST,
                    "dawn": RechargeType.DAWN,
                    "encounter": RechargeType.ENCOUNTER,
                    "at_will": RechargeType.AT_WILL,
                }
                recharge = recharge_map.get(resource_info.get("recharge", "at_will"), RechargeType.AT_WILL)
                
                tracked_features[feature_name.lower()] = Feature(
                    name=feature_name,
                    description=resource_info.get("description", ""),
                    uses_max=resource_info["uses_max"],
                    uses_current=resource_info["uses_max"],  # Start with full uses
                    recharge=recharge,
                    source="class" if feature_name in class_features_list else "race",
                    level_gained=level,
                )
            else:
                # At-will feature, just store the name
                simple_features.append(feature_name)
        
        # Build journal with personality/backstory using proper fields
        from dnd_manager.models.ecs import JournalComponent
        
        journal = JournalComponent(
            personality_traits=[t.strip() for t in personality_traits.split('\n') if t.strip()] if personality_traits else [],
            ideals=[ideals.strip()] if ideals and ideals.strip() else [],
            bonds=[bonds.strip()] if bonds and bonds.strip() else [],
            flaws=[flaws.strip()] if flaws and flaws.strip() else [],
            backstory=backstory.strip() if backstory else "",
            appearance=appearance.strip() if appearance else "",
            age=age.strip() if age else "",
            height=height.strip() if height else "",
            weight=weight.strip() if weight else "",
            eyes=eyes.strip() if eyes else "",
            hair=hair.strip() if hair else "",
            skin=skin.strip() if skin else "",
        )
        
        # Create the character
        # Build starting inventory first so we can calculate AC
        inventory = build_starting_inventory(char_class)
        
        # Calculate DEX modifier for AC
        dex_mod = (dexterity - 10) // 2
        
        # Calculate AC from equipped armor
        ac_base, max_dex, uses_dex = calculate_ac_from_inventory(inventory, dex_mod)
        
        # Build spellbook for caster classes
        from dnd_manager.models.progression import get_spellcasting_ability, SPELLCASTING_ABILITY
        
        spellbook = None
        if char_class in SPELLCASTING_ABILITY:
            # Get the spellcasting ability modifier
            ability = get_spellcasting_ability(char_class)
            if ability == "charisma":
                spell_mod = (charisma - 10) // 2
            elif ability == "wisdom":
                spell_mod = (wisdom - 10) // 2
            elif ability == "intelligence":
                spell_mod = (intelligence - 10) // 2
            else:
                spell_mod = 0
            
            spellbook = build_starting_spellbook(
                class_name=char_class,
                level=level,
                spellcasting_mod=spell_mod,
                proficiency_bonus=prof_bonus,
            )
        
        # Get racial speed
        racial_speed = racial_traits.get("speed", 30)
        
        character = ActorEntity(
            uid=uuid4(),
            name=name.strip(),
            type=ActorType.PLAYER,
            race=race,  # Keep full race name (with subrace)
            size=racial_traits.get("size", "Medium"),
            alignment=alignment,
            background=background,  # D&D background (Soldier, Noble, etc.)
            stats=StatsComponent(
                strength=strength,
                dexterity=dexterity,
                constitution=constitution,
                intelligence=intelligence,
                wisdom=wisdom,
                charisma=charisma,
                proficiency_bonus=prof_bonus,
                save_proficiencies=save_prof_enums,
                skill_proficiencies=skill_prof_dict,
            ),
            health=HealthComponent(
                hp_current=hp,
                hp_max=hp,
            ),
            defense=DefenseComponent(
                ac_base=ac_base,
                uses_dex=uses_dex,
                max_dex_bonus=max_dex,
            ),
            movement=MovementComponent(
                speed_walk=racial_speed,
                movement_remaining=racial_speed,
            ),
            class_features=ClassFeatureComponent(
                classes=[ClassLevel(class_name=char_class, level=level)],
                features=simple_features,
                tracked_features=tracked_features,
            ),
            inventory=inventory,
            spellbook=spellbook,
            journal=journal,
        )
        
        # Save
        st.session_state.characters[str(character.uid)] = character
        save_characters()
        
        st.session_state.char_message = ("success", f"Created **{character.name}**!")
        st.rerun()
    
    if not can_create:
        if not name.strip():
            st.caption("⚠️ Enter a character name to continue")
        elif not stats_valid:
            st.caption("⚠️ Fix ability score assignment to continue")


# =============================================================================
# Sidebar
# =============================================================================


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


# =============================================================================
# Entry Point
# =============================================================================


init_session_state()
render_sidebar()
main()
