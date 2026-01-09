"""Character creation wizard."""

from __future__ import annotations

import random
from uuid import uuid4

import streamlit as st

from dnd_manager.models.ecs import (
    Ability,
    ActorEntity,
    ActorType,
    ClassFeatureComponent,
    ClassLevel,
    DefenseComponent,
    Feature,
    HealthComponent,
    JournalComponent,
    MovementComponent,
    RechargeType,
    StatsComponent,
)

from .constants import (
    ALIGNMENTS,
    BACKGROUNDS,
    CLASSES,
    CLASS_HIT_DIE,
    CLASS_SAVE_PROFS,
    POINT_BUY_COSTS,
    POINT_BUY_TOTAL,
    RACES,
    STANDARD_ARRAY,
    STAT_METHODS,
)
from .state import (
    build_starting_inventory,
    build_starting_spellbook,
    calculate_ac_from_inventory,
    capture_creator_state,
    restore_creator_state,
    save_characters,
)


def roll_4d6_drop_lowest() -> int:
    """Roll 4d6 and drop the lowest die."""
    rolls = [random.randint(1, 6) for _ in range(4)]
    rolls.sort(reverse=True)
    return sum(rolls[:3])  # Sum top 3


def render_character_creator() -> None:
    """Render the character creation form."""
    
    st.markdown("### Create New Character")
    st.markdown("*Follow D&D 5e character creation rules*")
    
    # Undo/Redo controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 7])
    with col1:
        if st.button("↩ Undo", disabled=not st.session_state.cc_history.can_undo, key="undo_btn", help="Undo last change"):
            state = st.session_state.cc_history.undo()
            if state:
                restore_creator_state(state)
                st.rerun()
    with col2:
        if st.button("↪ Redo", disabled=not st.session_state.cc_history.can_redo, key="redo_btn", help="Redo last undone change"):
            state = st.session_state.cc_history.redo()
            if state:
                restore_creator_state(state)
                st.rerun()
    with col3:
        if st.button("📸 Save State", key="save_state_btn", help="Save current state to history"):
            current_state = capture_creator_state()
            st.session_state.cc_history.push(current_state)
            st.toast("State saved!", icon="✅")
    
    st.divider()
    
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
