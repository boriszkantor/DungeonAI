# DungeonAI

An AI Dungeon Master for D&D 5th Edition that uses your own rulebooks.

## What It Does

DungeonAI is a **solo D&D experience** where you chat with an AI Dungeon Master. You upload your rulebooks (PHB, DMG, Monster Manual, etc.) and adventures, create or import a character, and play through adventures in a chat interface.

The key difference from generic AI chatbots: **the AI cannot lie about mechanics**. Python handles all dice rolls, tracks your character state, and the AI only sees the results. When the AI needs to reference a rule, it searches your indexed rulebooks and cites the actual text.

## Core Features

**RAG-Powered Rules** — Import your D&D PDFs. The AI searches them for accurate rulings instead of making things up. Ask "how does grappling work?" and get the actual PHB text.

**Vision Character Import** — Upload a character sheet PDF. AI vision extracts your stats, features, spells, and equipment automatically.

**True Dice Rolls** — All rolls happen in Python using the d20 library. The AI cannot fudge numbers or decide you "just barely succeed" — what you roll is what you get.

**Knowledge Graph** — The system builds a graph of entity relationships from your rulebooks (which classes get which spells, what conditions affect what). This helps with complex lookups like "what resists fire damage?"

**Session Persistence** — Your adventures save locally. Resume exactly where you left off with full chat history.

## Installation

Requires Python 3.11+ and [Poppler](https://poppler.freedesktop.org/) for PDF processing.

```bash
git clone https://github.com/yourusername/DungeonAI.git
cd DungeonAI
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -e .
```

Create a `.env` file with your [OpenRouter](https://openrouter.ai) API key:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Run the app:

```bash
streamlit run src/dnd_manager/ui/app.py
```

## How to Use

1. **Library** — Upload your D&D rulebooks (PDF). The system chunks them, generates embeddings, and extracts entity relationships.

2. **Characters** — Import a character sheet PDF or create a character manually.

3. **Sessions** — Create a session linking your character to a campaign.

4. **Play** — Chat with the AI DM. Describe your actions, ask questions, roll dice.

## Architecture

The system separates **truth** from **narrative**:

- Python manages game state, dice, and rules enforcement
- AI handles storytelling and interprets actions  
- Neither can override the other

RAG retrieval uses a hybrid approach:
- Vector search (ChromaDB) for semantic matching
- Knowledge graph (NetworkX) for entity relationships
- HyDE (Hypothetical Document Embeddings) for natural language rule queries

```
src/dnd_manager/
├── core/        # Config, logging, exceptions
├── models/      # Pydantic schemas for game entities
├── ingestion/   # PDF processing, RAG, knowledge graph
├── dm/          # AI orchestration, memory management
├── storage/     # SQLite persistence
└── ui/          # Streamlit interface
```

## Requirements

- Python 3.11+
- OpenRouter API key (for AI features)
- Poppler (for PDF processing)
- ~4GB disk space for vector store

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src
mypy src
```

## License

MIT
