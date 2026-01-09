"""Session state management for character storage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import streamlit as st

from dnd_manager.core.logging import get_logger
from dnd_manager.ingestion.universal_loader import ChromaStore, UniversalIngestor
from dnd_manager.models.ecs import (
    Ability,
    ActorEntity,
    InventoryComponent,
    ItemType,
    SpellbookComponent,
)
from dnd_manager.models.equipment import get_starting_equipment
from dnd_manager.storage.database import get_database

logger = get_logger(__name__)


# =============================================================================
# Undo/Redo State History
# =============================================================================


@dataclass
class CharacterState:
    """Snapshot of character creation state."""
    name: str
    race: str
    char_class: str
    level: int
    alignment: str
    background: str
    stat_method: str
    stats: dict[str, int]  # ability -> value
    skills: list[str]
    personality_traits: str
    ideals: str
    bonds: str
    flaws: str
    backstory: str
    appearance: str
    age: str
    height: str
    weight: str
    eyes: str
    hair: str
    skin: str
    timestamp: datetime


class StateHistory:
    """Manages undo/redo stack for character creation."""
    
    def __init__(self, max_history: int = 50):
        self._history: list[CharacterState] = []
        self._current_index: int = -1
        self._max_history = max_history
    
    def push(self, state: CharacterState) -> None:
        """Add new state, truncating any redo history."""
        # Remove any states after current index (redo history)
        self._history = self._history[:self._current_index + 1]
        
        # Add new state
        self._history.append(state)
        self._current_index += 1
        
        # Enforce max history
        if len(self._history) > self._max_history:
            self._history.pop(0)
            self._current_index -= 1
    
    def undo(self) -> CharacterState | None:
        """Move back in history."""
        if self.can_undo:
            self._current_index -= 1
            return self._history[self._current_index]
        return None
    
    def redo(self) -> CharacterState | None:
        """Move forward in history."""
        if self.can_redo:
            self._current_index += 1
            return self._history[self._current_index]
        return None
    
    @property
    def can_undo(self) -> bool:
        return self._current_index > 0
    
    @property
    def can_redo(self) -> bool:
        return self._current_index < len(self._history) - 1


def capture_creator_state() -> CharacterState:
    """Capture current state of character creator from session state."""
    return CharacterState(
        name=st.session_state.get("cc_name", ""),
        race=st.session_state.get("cc_race", "Human"),
        char_class=st.session_state.get("cc_class", "Fighter"),
        level=st.session_state.get("cc_level", 1),
        alignment=st.session_state.get("cc_alignment", "True Neutral"),
        background=st.session_state.get("cc_background", "Acolyte"),
        stat_method=st.session_state.get("cc_stat_method", "standard_array"),
        stats={
            "strength": st.session_state.get("cc_str_sa", 10),
            "dexterity": st.session_state.get("cc_dex_sa", 10),
            "constitution": st.session_state.get("cc_con_sa", 10),
            "intelligence": st.session_state.get("cc_int_sa", 10),
            "wisdom": st.session_state.get("cc_wis_sa", 10),
            "charisma": st.session_state.get("cc_cha_sa", 10),
        },
        skills=st.session_state.get("cc_skills", []),
        personality_traits=st.session_state.get("cc_personality", ""),
        ideals=st.session_state.get("cc_ideals", ""),
        bonds=st.session_state.get("cc_bonds", ""),
        flaws=st.session_state.get("cc_flaws", ""),
        backstory=st.session_state.get("cc_backstory", ""),
        appearance=st.session_state.get("cc_appearance", ""),
        age=st.session_state.get("cc_age", ""),
        height=st.session_state.get("cc_height", ""),
        weight=st.session_state.get("cc_weight", ""),
        eyes=st.session_state.get("cc_eyes", ""),
        hair=st.session_state.get("cc_hair", ""),
        skin=st.session_state.get("cc_skin", ""),
        timestamp=datetime.now(),
    )


def restore_creator_state(state: CharacterState) -> None:
    """Restore character creator state to session state."""
    st.session_state.cc_name = state.name
    st.session_state.cc_race = state.race
    st.session_state.cc_class = state.char_class
    st.session_state.cc_level = state.level
    st.session_state.cc_alignment = state.alignment
    st.session_state.cc_background = state.background
    st.session_state.cc_stat_method = state.stat_method
    
    # Restore ability scores
    st.session_state.cc_str_sa = state.stats.get("strength", 10)
    st.session_state.cc_dex_sa = state.stats.get("dexterity", 10)
    st.session_state.cc_con_sa = state.stats.get("constitution", 10)
    st.session_state.cc_int_sa = state.stats.get("intelligence", 10)
    st.session_state.cc_wis_sa = state.stats.get("wisdom", 10)
    st.session_state.cc_cha_sa = state.stats.get("charisma", 10)
    
    # Restore other fields
    st.session_state.cc_skills = state.skills
    st.session_state.cc_personality = state.personality_traits
    st.session_state.cc_ideals = state.ideals
    st.session_state.cc_bonds = state.bonds
    st.session_state.cc_flaws = state.flaws
    st.session_state.cc_backstory = state.backstory
    st.session_state.cc_appearance = state.appearance
    st.session_state.cc_age = state.age
    st.session_state.cc_height = state.height
    st.session_state.cc_weight = state.weight
    st.session_state.cc_eyes = state.eyes
    st.session_state.cc_hair = state.hair
    st.session_state.cc_skin = state.skin


# =============================================================================
# Session State Management
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
    
    # Initialize undo/redo history for character creator
    if "cc_history" not in st.session_state:
        st.session_state.cc_history = StateHistory()


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


# =============================================================================
# Character Building Helpers
# =============================================================================


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
    )
    
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
