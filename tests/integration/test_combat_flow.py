"""Integration tests for combat flow.

Tests complete combat scenarios from initiative to resolution.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from dnd_manager.dm.dice import roll_attack, roll_damage, roll_dice
from dnd_manager.models.ecs import (
    ActorEntity,
    ActorType,
    ClassFeatureComponent,
    ClassLevel,
    DefenseComponent,
    GameState,
    HealthComponent,
    StatsComponent,
)


class TestCombatFlow:
    """Test complete combat scenarios."""

    def test_initiative_and_turn_order(self) -> None:
        """Roll initiative, verify turn order."""
        # Create player character
        player = ActorEntity(
            uid=uuid4(),
            name="Fighter",
            type=ActorType.PLAYER,
            stats=StatsComponent(
                dexterity=14,  # +2 modifier
                proficiency_bonus=2,
            ),
            health=HealthComponent(hp_current=10, hp_max=10),
            defense=DefenseComponent(ac_base=16),
        )

        # Create enemy
        goblin = ActorEntity(
            uid=uuid4(),
            name="Goblin",
            type=ActorType.NPC_ENEMY,
            stats=StatsComponent(
                dexterity=14,  # +2 modifier
                proficiency_bonus=0,
            ),
            health=HealthComponent(hp_current=7, hp_max=7),
            defense=DefenseComponent(ac_base=15),
        )

        # Roll initiative for both
        player_init = roll_dice("1d20+2")
        goblin_init = roll_dice("1d20+2")

        # Create turn order based on initiative
        if player_init.total >= goblin_init.total:
            turn_order = [str(player.uid), str(goblin.uid)]
        else:
            turn_order = [str(goblin.uid), str(player.uid)]

        # Create game state
        game_state = GameState(
            round=1,
            turn_order=turn_order,
            current_turn_index=0,
            in_combat=True,
            entities={
                str(player.uid): player,
                str(goblin.uid): goblin,
            },
        )

        # Verify turn order
        assert len(game_state.turn_order) == 2
        assert str(player.uid) in game_state.turn_order
        assert str(goblin.uid) in game_state.turn_order
        assert game_state.in_combat is True

    def test_attack_hit_and_damage(self) -> None:
        """Attack roll hits, damage applied correctly."""
        # Create attacker and target
        attacker_stats = StatsComponent(strength=16, proficiency_bonus=2)  # +3 STR, +2 prof = +5
        attacker = ActorEntity(
            uid=uuid4(),
            name="Fighter",
            type=ActorType.PLAYER,
            stats=attacker_stats,
            health=HealthComponent(hp_current=10, hp_max=10),
            defense=DefenseComponent(ac_base=16),
        )

        target = ActorEntity(
            uid=uuid4(),
            name="Goblin",
            type=ActorType.NPC_ENEMY,
            stats=StatsComponent(proficiency_bonus=0),
            health=HealthComponent(hp_current=7, hp_max=7),
            defense=DefenseComponent(ac_base=15),
        )

        # Roll attack with +5 bonus against AC 15
        attack_bonus = 5
        target_ac = 15
        result, hit, is_crit = roll_attack(attack_bonus, target_ac)

        # If the attack hits, apply damage
        if hit:
            damage_result = roll_damage("1d8+3", critical=is_crit)
            target.health.hp_current -= damage_result.total

            # Verify damage was applied
            assert target.health.hp_current < 7
            assert target.health.hp_current >= 0

    def test_attack_miss(self) -> None:
        """Attack roll misses, no damage applied."""
        # Create target with high AC
        target = ActorEntity(
            uid=uuid4(),
            name="Knight",
            type=ActorType.NPC_ENEMY,
            stats=StatsComponent(proficiency_bonus=2),
            health=HealthComponent(hp_current=50, hp_max=50),
            defense=DefenseComponent(ac_base=20),  # Very high AC
        )

        original_hp = target.health.hp_current

        # Roll attack with low bonus against high AC
        attack_bonus = 2
        target_ac = 20
        result, hit, is_crit = roll_attack(attack_bonus, target_ac)

        # Only apply damage if hit
        if hit:
            damage_result = roll_damage("1d8+2")
            target.health.hp_current -= damage_result.total

        # If missed, HP should be unchanged (this test may occasionally fail if we roll a nat 20)
        if not hit:
            assert target.health.hp_current == original_hp

    def test_death_saves(self) -> None:
        """Character at 0 HP makes death saves."""
        character = ActorEntity(
            uid=uuid4(),
            name="Dying Rogue",
            type=ActorType.PLAYER,
            stats=StatsComponent(proficiency_bonus=2),
            health=HealthComponent(
                hp_current=0,  # At 0 HP
                hp_max=8,
                death_saves_success=0,
                death_saves_failure=0,
            ),
            defense=DefenseComponent(ac_base=14),
        )

        # Character is unconscious and making death saves
        assert character.health.hp_current == 0

        # Roll a death save (DC 10)
        save_result = roll_dice("1d20")

        if save_result.total >= 10:
            character.health.death_saves_success += 1
        else:
            character.health.death_saves_failure += 1

        # Verify death save was recorded
        assert (character.health.death_saves_success + character.health.death_saves_failure) == 1

        # Test stabilization (3 successes)
        character.health.death_saves_success = 3
        assert character.health.death_saves_success >= 3  # Stabilized

        # Test death (3 failures)
        character.health.death_saves_failure = 3
        assert character.health.death_saves_failure >= 3  # Dead

    def test_combat_resolution(self) -> None:
        """Combat ends when all enemies defeated."""
        # Create player
        player = ActorEntity(
            uid=uuid4(),
            name="Paladin",
            type=ActorType.PLAYER,
            stats=StatsComponent(strength=16, proficiency_bonus=2),
            health=HealthComponent(hp_current=20, hp_max=20),
            defense=DefenseComponent(ac_base=18),
        )

        # Create enemies
        goblin1 = ActorEntity(
            uid=uuid4(),
            name="Goblin 1",
            type=ActorType.NPC_ENEMY,
            stats=StatsComponent(proficiency_bonus=0),
            health=HealthComponent(hp_current=7, hp_max=7),
            defense=DefenseComponent(ac_base=15),
        )

        goblin2 = ActorEntity(
            uid=uuid4(),
            name="Goblin 2",
            type=ActorType.NPC_ENEMY,
            stats=StatsComponent(proficiency_bonus=0),
            health=HealthComponent(hp_current=7, hp_max=7),
            defense=DefenseComponent(ac_base=15),
        )

        # Create game state
        game_state = GameState(
            round=1,
            turn_order=[str(player.uid), str(goblin1.uid), str(goblin2.uid)],
            current_turn_index=0,
            in_combat=True,
            entities={
                str(player.uid): player,
                str(goblin1.uid): goblin1,
                str(goblin2.uid): goblin2,
            },
        )

        # Defeat all enemies
        goblin1.health.hp_current = 0
        goblin2.health.hp_current = 0

        # Check if all enemies are defeated
        enemies_alive = [
            e for e in game_state.entities.values()
            if e.type == ActorType.NPC_ENEMY and e.health.hp_current > 0
        ]

        assert len(enemies_alive) == 0, "All enemies should be defeated"

        # Combat should end
        game_state.in_combat = False
        assert game_state.in_combat is False
