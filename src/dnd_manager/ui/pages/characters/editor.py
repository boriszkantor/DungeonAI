"""Character editor with comprehensive 5E sheet fields."""

from __future__ import annotations

import re

import streamlit as st

from dnd_manager.models.ecs import ActorEntity, ItemStack, SpellbookComponent

from .constants import CLASSES
from .state import save_characters


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
