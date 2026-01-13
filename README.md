<div align="center">

# 🐉 DungeonAI

### AI-Powered D&D 5th Edition Campaign Manager

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typed: mypy](https://img.shields.io/badge/typed-mypy-blue.svg)](https://mypy-lang.org/)

*A production-grade campaign management system combining RAG retrieval, knowledge graphs, and vision AI to deliver an intelligent digital Dungeon Master assistant.*

[Features](#-features) •
[Quick Start](#-quick-start) •
[Installation](#-installation) •
[Documentation](#-documentation) •
[Contributing](#-contributing)

</div>

---

## ✨ Features

### 🧠 Hybrid RAG + Knowledge Graph System

DungeonAI uses a **state-of-the-art hybrid retrieval system** that combines semantic search with structured knowledge graphs:

- **📚 PDF Ingestion** — Import your rulebooks (PHB, DMG, etc.) with automatic markdown conversion and semantic chunking
- **🔍 Vector Search** — ChromaDB-powered semantic search finds relevant rules and descriptions
- **🕸️ Knowledge Graph** — NetworkX graph captures entity relationships (spells → conditions, classes → features)
- **⚡ Hybrid Queries** — Combines vector similarity with graph traversal for "specific beats general" rule resolution

### 🤖 AI-Powered Dungeon Master

- **Vision AI Character Import** — Upload a character sheet PDF and watch AI extract all stats, features, and equipment
- **Contextual Rule Lookup** — AI responses cite specific rulebook passages with page references
- **Multi-hop Reasoning** — Knowledge graph enables complex queries like *"What affects a Monk's Evasion against Fireball?"*

### 🎮 Complete Game Management

| Feature | Description |
|---------|-------------|
| **Character Builder** | Full 5E character creation with intelligent Standard Array, racial features, and class abilities |
| **Combat Tracker** | Initiative with DEX tiebreakers, turn management, and action economy tracking |
| **Dice Engine** | Advanced roller supporting advantage, disadvantage, and 3 critical hit variants |
| **Campaign Persistence** | SQLite-backed sessions, scenes, and campaign state that persists across sessions |
| **PDF Export** | Generate fillable character sheet PDFs from your characters |

### 🎯 D&D 5E Rules Accuracy

- ✅ Warlock Pact Magic with proper short rest recovery
- ✅ Ability score caps (20 standard, 24 for level 20 Barbarians)
- ✅ Half-Elf flexible ability score choices
- ✅ Spell slot tracking per class with multiclass support
- ✅ Condition tracking with automatic effect application

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/dnd-campaign-manager.git
cd dnd-campaign-manager

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -e .

# Set up your API key (OpenRouter for AI features)
echo "OPENROUTER_API_KEY=your-key-here" > .env

# Launch the application
streamlit run src/dnd_manager/ui/app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Installation

### Prerequisites

- **Python 3.11+** — [Download Python](https://www.python.org/downloads/)
- **Poppler** — Required for PDF processing
  - **Windows**: Included in `poppler-25.12.0/` directory, or install via [Chocolatey](https://chocolatey.org/): `choco install poppler`
  - **macOS**: `brew install poppler`
  - **Linux**: `sudo apt-get install poppler-utils`

### Standard Installation

```bash
# Clone and enter the project
git clone https://github.com/yourusername/dnd-campaign-manager.git
cd dnd-campaign-manager

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install the package
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Configuration

Create a `.env` file in the project root:

```env
# Required for AI features (get a key at https://openrouter.ai)
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional: Override default models
OPENROUTER_VISION_MODEL=google/gemini-2.0-flash-001
OPENROUTER_REASONING_MODEL=google/gemini-2.0-flash-001

# Optional: Logging
DND_MANAGER_LOG_LEVEL=INFO
DND_MANAGER_DEBUG=false
```

---

## 🏗️ Architecture

DungeonAI follows a **Modular Monolith** architecture with strict domain separation:

```
src/dnd_manager/
├── core/                   # Foundation layer
│   ├── config.py           # Pydantic settings management
│   ├── constants.py        # Game constants (abilities, skills, etc.)
│   ├── exceptions.py       # Unified exception hierarchy
│   └── logging.py          # Structured logging with structlog
│
├── models/                 # Domain models (Pydantic V2)
│   ├── character.py        # Character, AbilityScores, Features
│   ├── combat.py           # CombatState, Initiative, Actions
│   ├── campaign.py         # Campaign, Session, Scene
│   └── entities.py         # Base entity types
│
├── ingestion/              # Document processing & RAG
│   ├── universal_loader.py # Unified ingestion pipeline
│   ├── pdf_parser.py       # PDF → Markdown conversion
│   ├── rag_store.py        # ChromaDB vector store
│   ├── knowledge_graph.py  # NetworkX graph operations
│   ├── triple_extractor.py # LLM-based relationship extraction
│   ├── entity_resolver.py  # Fuzzy entity matching
│   └── hybrid_retriever.py # Combined vector + graph search
│
├── engine/                 # Game logic
│   ├── dice.py             # Dice rolling & expressions
│   ├── turn_manager.py     # Combat turn sequencing
│   └── game_loop.py        # Session state machine
│
├── storage/                # Persistence
│   └── database.py         # SQLite operations
│
└── ui/                     # Streamlit interface
    ├── app.py              # Main application entry
    ├── theme.py            # D&D-themed styling
    ├── components/         # Reusable UI components
    └── pages/              # Application pages
        ├── 1_Library.py    # Rulebook management
        ├── 2_Characters.py # Character management
        ├── 3_Campaign.py   # Campaign management
        └── 4_Play.py       # Active gameplay
```

### Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   PDF       │────▶│   Markdown   │────▶│   Chunks        │
│   Rulebook  │     │   Converter  │     │   (by headers)  │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
              ┌─────────────────────┐                          ┌─────────────────────┐
              │   ChromaDB          │                          │   Triple Extractor  │
              │   (Vector Store)    │                          │   (LLM Pipeline)    │
              └──────────┬──────────┘                          └──────────┬──────────┘
                         │                                                 │
                         ▼                                                 ▼
              ┌─────────────────────┐                          ┌─────────────────────┐
              │   Semantic Search   │◀────── Hybrid Query ─────▶│   Knowledge Graph   │
              │   (Top-K retrieval) │                          │   (Graph traversal) │
              └─────────────────────┘                          └─────────────────────┘
```

---

## 📖 Documentation

### Importing Rulebooks

1. Navigate to **📚 Library** in the sidebar
2. Upload your D&D PDF (PHB, DMG, Monster Manual, etc.)
3. Select the document type and click **Index Document**
4. The system will:
   - Convert PDF to semantic markdown
   - Chunk content by section headers
   - Generate embeddings and store in ChromaDB
   - Extract entity relationships into the knowledge graph

### Character Creation

1. Navigate to **👤 Characters**
2. Choose **Create New** or **Import from PDF**
3. For PDF import: Upload a character sheet and AI will extract all data
4. For manual creation: Follow the guided wizard through:
   - Race selection (with ability score choices)
   - Class selection (with subclass at appropriate level)
   - Background selection
   - Ability score assignment (Standard Array or Point Buy)
   - Equipment and spell selection

### Running a Session

1. Navigate to **🎮 Play**
2. Select or create a campaign
3. Add characters to the session
4. Use the AI DM assistant to:
   - Look up rules with `@rules [query]`
   - Roll dice with standard notation (e.g., `2d6+3`)
   - Track combat with the initiative tracker
   - Manage conditions and effects

---

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/dnd_manager --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

### Code Quality

```bash
# Format code
ruff format src tests

# Lint code
ruff check src tests

# Type checking
mypy src

# Run all checks (pre-commit)
pre-commit run --all-files
```

### Project Standards

- **Type Hints**: Full type coverage with `mypy --strict`
- **Documentation**: Google-style docstrings
- **Testing**: 80%+ code coverage requirement
- **Formatting**: Enforced via `ruff format`
- **Linting**: Comprehensive rules via `ruff check`

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes with appropriate tests
4. **Ensure** all checks pass:
   ```bash
   ruff format src tests
   ruff check src tests
   mypy src
   pytest
   ```
5. **Commit** with a descriptive message: `git commit -m 'Add amazing feature'`
6. **Push** to your branch: `git push origin feature/amazing-feature`
7. **Open** a Pull Request

### Code Style

- Follow existing patterns in the codebase
- Add type hints to all functions
- Write docstrings for public APIs
- Include tests for new functionality

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Wizards of the Coast** for Dungeons & Dragons
- **OpenRouter** for unified LLM API access
- **ChromaDB** for vector storage
- **Streamlit** for the web framework

---

<div align="center">

**Built with ❤️ for the D&D community**

[Report Bug](https://github.com/yourusername/dnd-campaign-manager/issues) •
[Request Feature](https://github.com/yourusername/dnd-campaign-manager/issues)

</div>
