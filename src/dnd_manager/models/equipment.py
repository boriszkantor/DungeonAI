"""D&D 5E Equipment Data.

Contains weapon, armor, and starting equipment definitions
based on the Player's Handbook.
"""

from uuid import uuid4
from dnd_manager.models.ecs import ItemStack, ItemType


# =============================================================================
# Weapon Definitions (PHB Chapter 5)
# =============================================================================

WEAPONS = {
    # Simple Melee Weapons
    "club": {
        "name": "Club",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d4",
        "damage_type": "bludgeoning",
        "weight": 2.0,
        "value_cp": 10,
        "weapon_properties": ["light"],
    },
    "dagger": {
        "name": "Dagger",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d4",
        "damage_type": "piercing",
        "weight": 1.0,
        "value_cp": 200,
        "weapon_properties": ["finesse", "light", "thrown"],
        "range_normal": 20,
        "range_long": 60,
    },
    "greatclub": {
        "name": "Greatclub",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d8",
        "damage_type": "bludgeoning",
        "weight": 10.0,
        "value_cp": 20,
        "weapon_properties": ["two-handed"],
    },
    "handaxe": {
        "name": "Handaxe",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d6",
        "damage_type": "slashing",
        "weight": 2.0,
        "value_cp": 500,
        "weapon_properties": ["light", "thrown"],
        "range_normal": 20,
        "range_long": 60,
    },
    "javelin": {
        "name": "Javelin",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d6",
        "damage_type": "piercing",
        "weight": 2.0,
        "value_cp": 50,
        "weapon_properties": ["thrown"],
        "range_normal": 30,
        "range_long": 120,
    },
    "light_hammer": {
        "name": "Light Hammer",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d4",
        "damage_type": "bludgeoning",
        "weight": 2.0,
        "value_cp": 200,
        "weapon_properties": ["light", "thrown"],
        "range_normal": 20,
        "range_long": 60,
    },
    "mace": {
        "name": "Mace",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d6",
        "damage_type": "bludgeoning",
        "weight": 4.0,
        "value_cp": 500,
        "weapon_properties": [],
    },
    "quarterstaff": {
        "name": "Quarterstaff",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d6",
        "damage_type": "bludgeoning",
        "weight": 4.0,
        "value_cp": 20,
        "weapon_properties": ["versatile"],
        "properties": {"versatile_dice": "1d8"},
    },
    "sickle": {
        "name": "Sickle",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d4",
        "damage_type": "slashing",
        "weight": 2.0,
        "value_cp": 100,
        "weapon_properties": ["light"],
    },
    "spear": {
        "name": "Spear",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d6",
        "damage_type": "piercing",
        "weight": 3.0,
        "value_cp": 100,
        "weapon_properties": ["thrown", "versatile"],
        "range_normal": 20,
        "range_long": 60,
        "properties": {"versatile_dice": "1d8"},
    },
    
    # Simple Ranged Weapons
    "light_crossbow": {
        "name": "Light Crossbow",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d8",
        "damage_type": "piercing",
        "weight": 5.0,
        "value_cp": 2500,
        "weapon_properties": ["ammunition", "loading", "two-handed"],
        "range_normal": 80,
        "range_long": 320,
    },
    "shortbow": {
        "name": "Shortbow",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d6",
        "damage_type": "piercing",
        "weight": 2.0,
        "value_cp": 2500,
        "weapon_properties": ["ammunition", "two-handed"],
        "range_normal": 80,
        "range_long": 320,
    },
    "sling": {
        "name": "Sling",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d4",
        "damage_type": "bludgeoning",
        "weight": 0.0,
        "value_cp": 10,
        "weapon_properties": ["ammunition"],
        "range_normal": 30,
        "range_long": 120,
    },
    
    # Martial Melee Weapons
    "battleaxe": {
        "name": "Battleaxe",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d8",
        "damage_type": "slashing",
        "weight": 4.0,
        "value_cp": 1000,
        "weapon_properties": ["versatile"],
        "properties": {"versatile_dice": "1d10"},
    },
    "flail": {
        "name": "Flail",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d8",
        "damage_type": "bludgeoning",
        "weight": 2.0,
        "value_cp": 1000,
        "weapon_properties": [],
    },
    "glaive": {
        "name": "Glaive",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d10",
        "damage_type": "slashing",
        "weight": 6.0,
        "value_cp": 2000,
        "weapon_properties": ["heavy", "reach", "two-handed"],
    },
    "greataxe": {
        "name": "Greataxe",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d12",
        "damage_type": "slashing",
        "weight": 7.0,
        "value_cp": 3000,
        "weapon_properties": ["heavy", "two-handed"],
    },
    "greatsword": {
        "name": "Greatsword",
        "item_type": ItemType.WEAPON,
        "damage_dice": "2d6",
        "damage_type": "slashing",
        "weight": 6.0,
        "value_cp": 5000,
        "weapon_properties": ["heavy", "two-handed"],
    },
    "halberd": {
        "name": "Halberd",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d10",
        "damage_type": "slashing",
        "weight": 6.0,
        "value_cp": 2000,
        "weapon_properties": ["heavy", "reach", "two-handed"],
    },
    "lance": {
        "name": "Lance",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d12",
        "damage_type": "piercing",
        "weight": 6.0,
        "value_cp": 1000,
        "weapon_properties": ["reach", "special"],
    },
    "longsword": {
        "name": "Longsword",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d8",
        "damage_type": "slashing",
        "weight": 3.0,
        "value_cp": 1500,
        "weapon_properties": ["versatile"],
        "properties": {"versatile_dice": "1d10"},
    },
    "maul": {
        "name": "Maul",
        "item_type": ItemType.WEAPON,
        "damage_dice": "2d6",
        "damage_type": "bludgeoning",
        "weight": 10.0,
        "value_cp": 1000,
        "weapon_properties": ["heavy", "two-handed"],
    },
    "morningstar": {
        "name": "Morningstar",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d8",
        "damage_type": "piercing",
        "weight": 4.0,
        "value_cp": 1500,
        "weapon_properties": [],
    },
    "pike": {
        "name": "Pike",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d10",
        "damage_type": "piercing",
        "weight": 18.0,
        "value_cp": 500,
        "weapon_properties": ["heavy", "reach", "two-handed"],
    },
    "rapier": {
        "name": "Rapier",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d8",
        "damage_type": "piercing",
        "weight": 2.0,
        "value_cp": 2500,
        "weapon_properties": ["finesse"],
    },
    "scimitar": {
        "name": "Scimitar",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d6",
        "damage_type": "slashing",
        "weight": 3.0,
        "value_cp": 2500,
        "weapon_properties": ["finesse", "light"],
    },
    "shortsword": {
        "name": "Shortsword",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d6",
        "damage_type": "piercing",
        "weight": 2.0,
        "value_cp": 1000,
        "weapon_properties": ["finesse", "light"],
    },
    "trident": {
        "name": "Trident",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d6",
        "damage_type": "piercing",
        "weight": 4.0,
        "value_cp": 500,
        "weapon_properties": ["thrown", "versatile"],
        "range_normal": 20,
        "range_long": 60,
        "properties": {"versatile_dice": "1d8"},
    },
    "warhammer": {
        "name": "Warhammer",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d8",
        "damage_type": "bludgeoning",
        "weight": 2.0,
        "value_cp": 1500,
        "weapon_properties": ["versatile"],
        "properties": {"versatile_dice": "1d10"},
    },
    "war_pick": {
        "name": "War Pick",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d8",
        "damage_type": "piercing",
        "weight": 2.0,
        "value_cp": 500,
        "weapon_properties": [],
    },
    "whip": {
        "name": "Whip",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d4",
        "damage_type": "slashing",
        "weight": 3.0,
        "value_cp": 200,
        "weapon_properties": ["finesse", "reach"],
    },
    
    # Martial Ranged Weapons
    "blowgun": {
        "name": "Blowgun",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1",
        "damage_type": "piercing",
        "weight": 1.0,
        "value_cp": 1000,
        "weapon_properties": ["ammunition", "loading"],
        "range_normal": 25,
        "range_long": 100,
    },
    "hand_crossbow": {
        "name": "Hand Crossbow",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d6",
        "damage_type": "piercing",
        "weight": 3.0,
        "value_cp": 7500,
        "weapon_properties": ["ammunition", "light", "loading"],
        "range_normal": 30,
        "range_long": 120,
    },
    "heavy_crossbow": {
        "name": "Heavy Crossbow",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d10",
        "damage_type": "piercing",
        "weight": 18.0,
        "value_cp": 5000,
        "weapon_properties": ["ammunition", "heavy", "loading", "two-handed"],
        "range_normal": 100,
        "range_long": 400,
    },
    "longbow": {
        "name": "Longbow",
        "item_type": ItemType.WEAPON,
        "damage_dice": "1d8",
        "damage_type": "piercing",
        "weight": 2.0,
        "value_cp": 5000,
        "weapon_properties": ["ammunition", "heavy", "two-handed"],
        "range_normal": 150,
        "range_long": 600,
    },
    "net": {
        "name": "Net",
        "item_type": ItemType.WEAPON,
        "damage_dice": "0",
        "damage_type": "none",
        "weight": 3.0,
        "value_cp": 100,
        "weapon_properties": ["special", "thrown"],
        "range_normal": 5,
        "range_long": 15,
    },
}

# =============================================================================
# Armor Definitions (PHB Chapter 5)
# =============================================================================

ARMOR = {
    # Light Armor
    "padded": {
        "name": "Padded Armor",
        "item_type": ItemType.ARMOR,
        "ac_base": 11,
        "max_dex_bonus": None,  # No limit
        "stealth_disadvantage": True,
        "weight": 8.0,
        "value_cp": 500,
    },
    "leather": {
        "name": "Leather Armor",
        "item_type": ItemType.ARMOR,
        "ac_base": 11,
        "max_dex_bonus": None,
        "stealth_disadvantage": False,
        "weight": 10.0,
        "value_cp": 1000,
    },
    "studded_leather": {
        "name": "Studded Leather Armor",
        "item_type": ItemType.ARMOR,
        "ac_base": 12,
        "max_dex_bonus": None,
        "stealth_disadvantage": False,
        "weight": 13.0,
        "value_cp": 4500,
    },
    
    # Medium Armor
    "hide": {
        "name": "Hide Armor",
        "item_type": ItemType.ARMOR,
        "ac_base": 12,
        "max_dex_bonus": 2,
        "stealth_disadvantage": False,
        "weight": 12.0,
        "value_cp": 1000,
    },
    "chain_shirt": {
        "name": "Chain Shirt",
        "item_type": ItemType.ARMOR,
        "ac_base": 13,
        "max_dex_bonus": 2,
        "stealth_disadvantage": False,
        "weight": 20.0,
        "value_cp": 5000,
    },
    "scale_mail": {
        "name": "Scale Mail",
        "item_type": ItemType.ARMOR,
        "ac_base": 14,
        "max_dex_bonus": 2,
        "stealth_disadvantage": True,
        "weight": 45.0,
        "value_cp": 5000,
    },
    "breastplate": {
        "name": "Breastplate",
        "item_type": ItemType.ARMOR,
        "ac_base": 14,
        "max_dex_bonus": 2,
        "stealth_disadvantage": False,
        "weight": 20.0,
        "value_cp": 40000,
    },
    "half_plate": {
        "name": "Half Plate",
        "item_type": ItemType.ARMOR,
        "ac_base": 15,
        "max_dex_bonus": 2,
        "stealth_disadvantage": True,
        "weight": 40.0,
        "value_cp": 75000,
    },
    
    # Heavy Armor
    "ring_mail": {
        "name": "Ring Mail",
        "item_type": ItemType.ARMOR,
        "ac_base": 14,
        "max_dex_bonus": 0,
        "stealth_disadvantage": True,
        "weight": 40.0,
        "value_cp": 3000,
    },
    "chain_mail": {
        "name": "Chain Mail",
        "item_type": ItemType.ARMOR,
        "ac_base": 16,
        "max_dex_bonus": 0,
        "stealth_disadvantage": True,
        "strength_requirement": 13,
        "weight": 55.0,
        "value_cp": 7500,
    },
    "splint": {
        "name": "Splint Armor",
        "item_type": ItemType.ARMOR,
        "ac_base": 17,
        "max_dex_bonus": 0,
        "stealth_disadvantage": True,
        "strength_requirement": 15,
        "weight": 60.0,
        "value_cp": 20000,
    },
    "plate": {
        "name": "Plate Armor",
        "item_type": ItemType.ARMOR,
        "ac_base": 18,
        "max_dex_bonus": 0,
        "stealth_disadvantage": True,
        "strength_requirement": 15,
        "weight": 65.0,
        "value_cp": 150000,
    },
    
    # Shield
    "shield": {
        "name": "Shield",
        "item_type": ItemType.SHIELD,
        "ac_bonus": 2,
        "weight": 6.0,
        "value_cp": 1000,
    },
}

# =============================================================================
# Gear and Packs
# =============================================================================

GEAR = {
    "arrows_20": {
        "name": "Arrows (20)",
        "item_type": ItemType.GEAR,
        "quantity": 20,
        "weight": 1.0,
        "value_cp": 100,
    },
    "bolts_20": {
        "name": "Crossbow Bolts (20)",
        "item_type": ItemType.GEAR,
        "quantity": 20,
        "weight": 1.5,
        "value_cp": 100,
    },
    "backpack": {
        "name": "Backpack",
        "item_type": ItemType.GEAR,
        "weight": 5.0,
        "value_cp": 200,
    },
    "bedroll": {
        "name": "Bedroll",
        "item_type": ItemType.GEAR,
        "weight": 7.0,
        "value_cp": 100,
    },
    "component_pouch": {
        "name": "Component Pouch",
        "item_type": ItemType.GEAR,
        "weight": 2.0,
        "value_cp": 2500,
    },
    "arcane_focus": {
        "name": "Arcane Focus (Crystal)",
        "item_type": ItemType.GEAR,
        "weight": 1.0,
        "value_cp": 1000,
    },
    "holy_symbol": {
        "name": "Holy Symbol (Amulet)",
        "item_type": ItemType.GEAR,
        "weight": 1.0,
        "value_cp": 500,
    },
    "druidic_focus": {
        "name": "Druidic Focus (Wooden Staff)",
        "item_type": ItemType.GEAR,
        "weight": 4.0,
        "value_cp": 500,
    },
    "thieves_tools": {
        "name": "Thieves' Tools",
        "item_type": ItemType.TOOL,
        "weight": 1.0,
        "value_cp": 2500,
    },
    "musical_instrument": {
        "name": "Lute",
        "item_type": ItemType.TOOL,
        "weight": 2.0,
        "value_cp": 3500,
    },
    "rations_1day": {
        "name": "Rations (1 day)",
        "item_type": ItemType.CONSUMABLE,
        "weight": 2.0,
        "value_cp": 50,
        "is_stackable": True,
    },
    "rope_50ft": {
        "name": "Rope, Hempen (50 ft)",
        "item_type": ItemType.GEAR,
        "weight": 10.0,
        "value_cp": 100,
    },
    "torch": {
        "name": "Torch",
        "item_type": ItemType.GEAR,
        "weight": 1.0,
        "value_cp": 1,
        "is_stackable": True,
    },
    "waterskin": {
        "name": "Waterskin",
        "item_type": ItemType.GEAR,
        "weight": 5.0,
        "value_cp": 20,
    },
}


# =============================================================================
# Starting Equipment by Class (PHB)
# =============================================================================

# Default starting equipment - simplified choices (Option A for most)
CLASS_STARTING_EQUIPMENT = {
    "Barbarian": {
        "weapons": ["greataxe", "handaxe", "handaxe", "javelin", "javelin", "javelin", "javelin"],
        "armor": [],
        "gear": ["backpack", "bedroll", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "rope_50ft", "torch", "torch", "torch", "torch", "torch", "torch", "torch", "torch", "torch", "torch", "waterskin"],
        "gold_cp": 1000,  # 10 gp
        "equipped_weapon": "greataxe",
        "equipped_armor": None,
    },
    "Bard": {
        "weapons": ["rapier", "dagger"],
        "armor": ["leather"],
        "gear": ["backpack", "bedroll", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "waterskin", "musical_instrument"],
        "gold_cp": 1000,
        "equipped_weapon": "rapier",
        "equipped_armor": "leather",
    },
    "Cleric": {
        "weapons": ["mace", "light_crossbow"],
        "armor": ["scale_mail", "shield"],
        "gear": ["backpack", "bedroll", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "waterskin", "holy_symbol", "bolts_20"],
        "gold_cp": 1000,
        "equipped_weapon": "mace",
        "equipped_armor": "scale_mail",
    },
    "Druid": {
        "weapons": ["scimitar"],
        "armor": ["leather", "shield"],
        "gear": ["backpack", "bedroll", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "waterskin", "druidic_focus"],
        "gold_cp": 1000,
        "equipped_weapon": "scimitar",
        "equipped_armor": "leather",
    },
    "Fighter": {
        "weapons": ["longsword", "longbow"],
        "armor": ["chain_mail", "shield"],
        "gear": ["backpack", "bedroll", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "rope_50ft", "waterskin", "arrows_20"],
        "gold_cp": 1000,
        "equipped_weapon": "longsword",
        "equipped_armor": "chain_mail",
    },
    "Monk": {
        "weapons": ["shortsword", "dart", "dart", "dart", "dart", "dart", "dart", "dart", "dart", "dart", "dart"],
        "armor": [],
        "gear": ["backpack", "bedroll", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "waterskin"],
        "gold_cp": 500,
        "equipped_weapon": "shortsword",
        "equipped_armor": None,
    },
    "Paladin": {
        "weapons": ["longsword", "javelin", "javelin", "javelin", "javelin", "javelin"],
        "armor": ["chain_mail", "shield"],
        "gear": ["backpack", "bedroll", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "waterskin", "holy_symbol"],
        "gold_cp": 1000,
        "equipped_weapon": "longsword",
        "equipped_armor": "chain_mail",
    },
    "Ranger": {
        "weapons": ["shortsword", "shortsword", "longbow"],
        "armor": ["scale_mail"],
        "gear": ["backpack", "bedroll", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "rope_50ft", "waterskin", "arrows_20"],
        "gold_cp": 1000,
        "equipped_weapon": "longbow",
        "equipped_armor": "scale_mail",
    },
    "Rogue": {
        "weapons": ["rapier", "shortbow", "dagger", "dagger"],
        "armor": ["leather"],
        "gear": ["backpack", "bedroll", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "waterskin", "thieves_tools", "arrows_20"],
        "gold_cp": 1000,
        "equipped_weapon": "rapier",
        "equipped_armor": "leather",
    },
    "Sorcerer": {
        "weapons": ["light_crossbow", "dagger", "dagger"],
        "armor": [],
        "gear": ["backpack", "bedroll", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "waterskin", "component_pouch", "bolts_20"],
        "gold_cp": 1000,
        "equipped_weapon": "light_crossbow",
        "equipped_armor": None,
    },
    "Warlock": {
        "weapons": ["light_crossbow", "dagger", "dagger"],
        "armor": ["leather"],
        "gear": ["backpack", "bedroll", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "waterskin", "component_pouch", "bolts_20"],
        "gold_cp": 1000,
        "equipped_weapon": "light_crossbow",
        "equipped_armor": "leather",
    },
    "Wizard": {
        "weapons": ["quarterstaff", "dagger"],
        "armor": [],
        "gear": ["backpack", "bedroll", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "rations_1day", "waterskin", "component_pouch", "arcane_focus"],
        "gold_cp": 1000,
        "equipped_weapon": "quarterstaff",
        "equipped_armor": None,
    },
}


# =============================================================================
# Factory Functions
# =============================================================================

def create_item(item_id: str, equipped: bool = False) -> ItemStack | None:
    """Create an ItemStack from an item ID.
    
    Args:
        item_id: The ID of the item (e.g., 'longsword', 'leather').
        equipped: Whether the item should be equipped.
        
    Returns:
        ItemStack or None if item not found.
    """
    # Check all item categories
    item_data = None
    if item_id in WEAPONS:
        item_data = WEAPONS[item_id].copy()
    elif item_id in ARMOR:
        item_data = ARMOR[item_id].copy()
    elif item_id in GEAR:
        item_data = GEAR[item_id].copy()
    
    if item_data is None:
        return None
    
    # Create the ItemStack
    return ItemStack(
        uid=uuid4(),
        item_id=item_id,
        equipped=equipped,
        **item_data,
    )


def get_starting_equipment(class_name: str) -> tuple[list[ItemStack], int]:
    """Get starting equipment for a class.
    
    Args:
        class_name: Character class name.
        
    Returns:
        Tuple of (list of ItemStack, starting gold in copper).
    """
    equipment_data = CLASS_STARTING_EQUIPMENT.get(class_name)
    if equipment_data is None:
        # Default equipment for unknown class
        return [create_item("dagger", equipped=True)], 1000
    
    items = []
    equipped_weapon = equipment_data.get("equipped_weapon")
    equipped_armor = equipment_data.get("equipped_armor")
    
    # Add weapons
    for weapon_id in equipment_data.get("weapons", []):
        is_equipped = (weapon_id == equipped_weapon and not any(
            i.item_id == weapon_id and i.equipped for i in items
        ))
        item = create_item(weapon_id, equipped=is_equipped)
        if item:
            items.append(item)
    
    # Add armor
    for armor_id in equipment_data.get("armor", []):
        is_equipped = (armor_id == equipped_armor and not any(
            i.item_id == armor_id and i.equipped for i in items
        ))
        item = create_item(armor_id, equipped=is_equipped)
        if item:
            items.append(item)
    
    # Add gear
    for gear_id in equipment_data.get("gear", []):
        item = create_item(gear_id, equipped=False)
        if item:
            items.append(item)
    
    gold_cp = equipment_data.get("gold_cp", 1000)
    
    return items, gold_cp


def resolve_item_name(item_name: str) -> ItemStack:
    """Resolve an item name to an ItemStack, with fuzzy matching.
    
    Tries to match the item name to known items, handling variations
    like "Longsword +1" -> base longsword with magic bonus.
    
    Args:
        item_name: The item name from a character sheet.
        
    Returns:
        ItemStack with best-match stats.
    """
    import re
    
    original_name = item_name
    name_lower = item_name.lower().strip()
    
    # Check for magic item modifiers (+1, +2, +3)
    magic_bonus = 0
    magic_match = re.search(r'\+(\d)', name_lower)
    if magic_match:
        magic_bonus = int(magic_match.group(1))
        name_lower = re.sub(r'\s*\+\d\s*', ' ', name_lower).strip()
    
    # Normalize common variations
    name_normalized = (
        name_lower
        .replace("'s", "")
        .replace("'", "")
        .replace("-", "_")
        .replace(" ", "_")
    )
    
    # Try direct match
    if name_normalized in WEAPONS:
        item = create_item(name_normalized)
        if item and magic_bonus:
            item.name = original_name  # Keep original name
            item.properties["magic_bonus"] = magic_bonus
        return item
    
    if name_normalized in ARMOR:
        item = create_item(name_normalized)
        if item and magic_bonus:
            item.name = original_name
            item.properties["magic_bonus"] = magic_bonus
            if item.ac_base:
                item.ac_base += magic_bonus
        return item
    
    if name_normalized in GEAR:
        return create_item(name_normalized)
    
    # Try fuzzy matching - check if any known item name is contained
    for item_id in WEAPONS:
        weapon_name = WEAPONS[item_id]["name"].lower()
        if weapon_name in name_lower or name_lower in weapon_name:
            item = create_item(item_id)
            if item:
                item.name = original_name
                if magic_bonus:
                    item.properties["magic_bonus"] = magic_bonus
                return item
    
    for item_id in ARMOR:
        armor_name = ARMOR[item_id]["name"].lower()
        if armor_name in name_lower or name_lower in armor_name:
            item = create_item(item_id)
            if item:
                item.name = original_name
                if magic_bonus:
                    item.properties["magic_bonus"] = magic_bonus
                    if item.ac_base:
                        item.ac_base += magic_bonus
                return item
    
    # Check for armor type keywords
    armor_keywords = {
        "plate": "plate",
        "chain mail": "chain_mail",
        "chain shirt": "chain_shirt",
        "leather": "leather",
        "studded": "studded_leather",
        "scale": "scale_mail",
        "half plate": "half_plate",
        "breastplate": "breastplate",
        "hide": "hide",
        "ring mail": "ring_mail",
        "splint": "splint",
        "padded": "padded",
        "shield": "shield",
    }
    
    for keyword, item_id in armor_keywords.items():
        if keyword in name_lower:
            item = create_item(item_id)
            if item:
                item.name = original_name
                if magic_bonus:
                    item.properties["magic_bonus"] = magic_bonus
                    if item.ac_base:
                        item.ac_base += magic_bonus
                return item
    
    # Weapon type keywords
    weapon_keywords = {
        "sword": "longsword",
        "dagger": "dagger",
        "bow": "shortbow",
        "longbow": "longbow",
        "crossbow": "light_crossbow",
        "heavy crossbow": "heavy_crossbow",
        "axe": "battleaxe",
        "greataxe": "greataxe",
        "hammer": "warhammer",
        "mace": "mace",
        "staff": "quarterstaff",
        "rapier": "rapier",
        "scimitar": "scimitar",
        "spear": "spear",
        "javelin": "javelin",
        "flail": "flail",
        "morningstar": "morningstar",
        "glaive": "glaive",
        "halberd": "halberd",
        "pike": "pike",
        "trident": "trident",
        "whip": "whip",
        "club": "club",
        "sickle": "sickle",
    }
    
    for keyword, item_id in weapon_keywords.items():
        if keyword in name_lower:
            item = create_item(item_id)
            if item:
                item.name = original_name
                if magic_bonus:
                    item.properties["magic_bonus"] = magic_bonus
                return item
    
    # Unknown item - create generic
    return ItemStack(
        uid=uuid4(),
        item_id=name_normalized,
        name=original_name,
        item_type=ItemType.GEAR,
        quantity=1,
        description=f"Unknown item: {original_name}",
    )


def resolve_item_from_rag(item_name: str, chroma_store) -> ItemStack | None:
    """Try to resolve an item using RAG search.
    
    Searches indexed sourcebooks for item stats.
    
    Args:
        item_name: Item name to search for.
        chroma_store: ChromaStore instance.
        
    Returns:
        ItemStack if found, None otherwise.
    """
    import re
    
    try:
        # Search for the item
        results = chroma_store.search(
            f"{item_name} item stats properties",
            n_results=3,
        )
        
        if not results:
            return None
        
        # Try to parse item stats from the content
        content = results[0].content.lower()
        
        # Look for damage dice pattern
        damage_match = re.search(r'(\d+d\d+)', content)
        damage_dice = damage_match.group(1) if damage_match else None
        
        # Look for damage type
        damage_types = ["slashing", "piercing", "bludgeoning", "fire", "cold", 
                       "lightning", "thunder", "poison", "acid", "necrotic", 
                       "radiant", "force", "psychic"]
        damage_type = None
        for dt in damage_types:
            if dt in content:
                damage_type = dt
                break
        
        # Look for AC
        ac_match = re.search(r'ac[:\s]+(\d+)', content)
        ac_base = int(ac_match.group(1)) if ac_match else None
        
        # Determine item type
        item_type = ItemType.GEAR
        if damage_dice:
            item_type = ItemType.WEAPON
        elif ac_base:
            item_type = ItemType.ARMOR
        
        return ItemStack(
            uid=uuid4(),
            item_id=item_name.lower().replace(" ", "_"),
            name=item_name,
            item_type=item_type,
            damage_dice=damage_dice,
            damage_type=damage_type,
            ac_base=ac_base,
            description=f"From sourcebook: {results[0].source}",
        )
        
    except Exception:
        return None


__all__ = [
    "WEAPONS",
    "ARMOR",
    "GEAR",
    "CLASS_STARTING_EQUIPMENT",
    "create_item",
    "get_starting_equipment",
    "resolve_item_name",
    "resolve_item_from_rag",
]
