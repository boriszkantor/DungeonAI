# D&D 5E AI Campaign Manager

A production-grade, AI-powered campaign management system for Dungeons & Dragons 5th Edition, built with a modular monolith architecture.

## Features

- 🎲 **Dice Rolling Engine** - Full D&D 5E dice mechanics with advantage, disadvantage, and critical hits
- ⚔️ **Combat Tracker** - Initiative tracking, turn management, and action economy
- 📖 **Campaign Management** - Scenes, sessions, and campaign state persistence
- 🤖 **AI Integration** - Gemini and OpenAI support for DM assistance
- 📚 **RAG Pipeline** - PDF ingestion and semantic search for rulebooks
- 🖥️ **Streamlit UI** - Modern web interface for campaign management

## Architecture

This project follows a **Modular Monolith** pattern with strict domain separation:

```
src/dnd_manager/
├── core/           # Configuration, logging, base exceptions
├── models/         # Pydantic V2 schemas for game entities
├── ingestion/      # PDF parsing, OCR, RAG pipeline
├── engine/         # Game loop, turn management, dice logic
└── ui/             # Streamlit interface
```

## Requirements

- Python 3.11+
- Dependencies managed via `pyproject.toml`

## Installation

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/dnd-manager.git
cd dnd-manager
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Unix/macOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

4. Copy the environment template and configure:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application

```bash
streamlit run src/dnd_manager/ui/app.py
```

Or using the installed entry point:
```bash
dnd-manager
```

## Configuration

Configuration is managed through environment variables or a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `DND_MANAGER_GEMINI_API_KEY` | Google Gemini API key | Required if using Gemini |
| `DND_MANAGER_OPENAI_API_KEY` | OpenAI API key | Required if using OpenAI |
| `DND_MANAGER_DEFAULT_PROVIDER` | Default AI provider | `gemini` |
| `DND_MANAGER_LOG_LEVEL` | Logging level | `INFO` |
| `DND_MANAGER_DEBUG` | Enable debug mode | `false` |

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/dnd_manager --cov-report=html

# Run specific test file
pytest tests/unit/engine/test_dice.py

# Run tests matching a pattern
pytest -k "test_roll"
```

### Code Quality

```bash
# Format code
ruff format src tests

# Lint code
ruff check src tests

# Type checking
mypy src
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Project Structure

```
D&D Campaign Manager/
├── src/
│   └── dnd_manager/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py          # Pydantic settings
│       │   ├── exceptions.py      # Exception hierarchy
│       │   └── logging.py         # Structured logging
│       ├── models/
│       │   ├── __init__.py
│       │   ├── character.py       # Character schemas
│       │   ├── combat.py          # Combat schemas
│       │   └── campaign.py        # Campaign schemas
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── pdf_parser.py      # PDF extraction
│       │   ├── ocr.py             # Image OCR
│       │   └── rag_pipeline.py    # Vector store & retrieval
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── dice.py            # Dice rolling
│       │   ├── turn_manager.py    # Combat turns
│       │   └── game_loop.py       # Game state machine
│       └── ui/
│           ├── __init__.py
│           ├── app.py             # Main Streamlit app
│           └── components.py      # UI components
├── tests/
│   ├── conftest.py
│   ├── unit/
│   └── integration/
├── docs/
├── data/
│   ├── scenes/
│   ├── assets/
│   └── cache/
├── pyproject.toml
├── README.md
└── .env.example
```

## Exception Handling

All exceptions inherit from `DndManagerError` for unified error handling:

```python
from dnd_manager.core.exceptions import DiceRollError

try:
    result = roll("invalid")
except DiceRollError as e:
    print(f"Error: {e.message}")
    print(f"Details: {e.details}")
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- All tests pass
- Code is formatted with `ruff format`
- No linting errors with `ruff check`
- Type hints are complete and pass `mypy --strict`
