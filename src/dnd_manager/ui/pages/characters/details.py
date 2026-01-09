"""Character details view rendering."""

from __future__ import annotations

import streamlit as st

from dnd_manager.models.ecs import Ability, ActorEntity


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
