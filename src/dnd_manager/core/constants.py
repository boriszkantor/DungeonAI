"""Application-wide constants for the D&D 5E AI Campaign Manager.

This module defines constants used throughout the application,
including display limits, D&D 5E rules constants, and configuration values.
"""

from __future__ import annotations

# =============================================================================
# Display Limits
# =============================================================================

MAX_DISPLAYED_ITEMS = 15
"""Maximum number of inventory items to display before truncating."""

MAX_DISPLAYED_FEATURES = 10
"""Maximum number of features to display before truncating."""

MAX_STORED_MEMORIES = 100
"""Maximum number of memories to store in a character's journal."""

MESSAGES_PER_PAGE = 50
"""Number of chat messages to display per page before pagination."""

# =============================================================================
# D&D 5E Rules Constants
# =============================================================================

PC_ABILITY_SCORE_CAP = 20
"""Maximum ability score for player characters (RAW D&D 5E)."""

BARBARIAN_CAPSTONE_CAP = 24
"""Maximum STR/CON for Barbarian at level 20 (Primal Champion)."""

MONSTER_ABILITY_SCORE_CAP = 30
"""Maximum ability score for monsters (RAW D&D 5E)."""

MIN_ABILITY_SCORE = 1
"""Minimum ability score (1 is barely functioning)."""

# =============================================================================
# Point Buy Constants (PHB p.13)
# =============================================================================

POINT_BUY_TOTAL = 27
"""Total points available for point buy character creation."""

POINT_BUY_MIN = 8
"""Minimum ability score in point buy."""

POINT_BUY_MAX = 15
"""Maximum ability score in point buy (before racial bonuses)."""

# Point buy costs for each score
POINT_BUY_COSTS = {
    8: 0,
    9: 1,
    10: 2,
    11: 3,
    12: 4,
    13: 5,
    14: 7,
    15: 9,
}

# =============================================================================
# Standard Array (PHB p.13)
# =============================================================================

STANDARD_ARRAY = [15, 14, 13, 12, 10, 8]
"""Standard array values for ability scores."""

# =============================================================================
# Combat Constants
# =============================================================================

DEFAULT_SPEED = 30
"""Default walking speed in feet (most medium creatures)."""

DEFAULT_INITIATIVE_DEX_MOD = 0
"""Default DEX modifier for initiative if stats unavailable."""

MAX_DEATH_SAVES = 3
"""Maximum death saving throws (3 successes = stable, 3 failures = dead)."""

# =============================================================================
# Miscellaneous
# =============================================================================

MAX_CHARACTER_LEVEL = 20
"""Maximum character level in D&D 5E."""

MIN_CHARACTER_LEVEL = 1
"""Minimum character level."""

DEFAULT_PROFICIENCY_BONUS = 2
"""Default proficiency bonus for level 1 characters."""


__all__ = [
    # Display
    "MAX_DISPLAYED_ITEMS",
    "MAX_DISPLAYED_FEATURES",
    "MAX_STORED_MEMORIES",
    "MESSAGES_PER_PAGE",
    # Ability Scores
    "PC_ABILITY_SCORE_CAP",
    "BARBARIAN_CAPSTONE_CAP",
    "MONSTER_ABILITY_SCORE_CAP",
    "MIN_ABILITY_SCORE",
    # Point Buy
    "POINT_BUY_TOTAL",
    "POINT_BUY_MIN",
    "POINT_BUY_MAX",
    "POINT_BUY_COSTS",
    # Standard Array
    "STANDARD_ARRAY",
    # Combat
    "DEFAULT_SPEED",
    "DEFAULT_INITIATIVE_DEX_MOD",
    "MAX_DEATH_SAVES",
    # Misc
    "MAX_CHARACTER_LEVEL",
    "MIN_CHARACTER_LEVEL",
    "DEFAULT_PROFICIENCY_BONUS",
]
