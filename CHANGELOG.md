# Changelog

All notable changes to the D&D 5E AI Campaign Manager will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Constants Module**: New `core/constants.py` with application-wide constants for D&D 5E rules, display limits, and point buy values
- **PDF Export**: Character sheet PDF export functionality using reportlab
- **Dice Animations**: CSS animations for dice rolls with special effects for natural 20s and natural 1s
- **Chat Pagination**: Long chat histories now paginate at 50 messages for better performance
- **Warlock Pact Magic**: Full support for Warlock pact magic slots with short rest recovery
- **Critical Hit Variants**: All three critical hit calculation methods:
  - `double_dice`: Double the number of dice (RAW D&D 5E)
  - `double_damage`: Roll normal damage then double the total
  - `max_plus_roll`: Maximum damage + normal roll
- **Half-Elf Ability Choice**: UI for selecting +1 to two abilities (in addition to +2 Charisma)
- **Smart Standard Array**: Intelligent selection that prevents duplicate value assignments
- **Ability Score Caps**: Automatic validation of PC ability scores (20 cap, 24 for Barbarian level 20)

### Changed
- **Model Consolidation**: Unified three parallel character model systems into ECS architecture
  - `models/character.py`, `models/entities.py`, `models/components.py` now deprecated with backward compatibility
  - `models/ecs.py` is the single source of truth for entity models
- **Initiative System**: TurnManager now correctly uses combatant's actual DEX modifier instead of hardcoded 0
- **Test Fixtures**: Updated `tests/conftest.py` to use `ActorEntity` from ECS system

### Fixed
- **Character Creator**: Fixed undefined variable bug where `final_str`, `final_dex`, etc. were referenced without being defined
- **Standard Array**: Prevented users from assigning the same value to multiple abilities

### Deprecated
- `models/character.py`: Use `models/ecs.ActorEntity` instead
- `models/entities.py`: Use `models/ecs.ActorEntity` instead  
- `models/components.py`: Import components from `models/ecs` instead

## [0.1.0] - Initial Release

### Added
- Basic D&D 5E campaign management
- AI-powered Dungeon Master
- Character creation and management
- Combat system with initiative tracking
- RAG-based rulebook queries
- Vision AI for character sheet extraction
- Session persistence with SQLite
