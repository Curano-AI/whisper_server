# Technology Stack

## Core Framework & Libraries

- **FastAPI**: Modern async web framework for building APIs
- **WhisperX**: Enhanced Whisper implementation with speaker diarization and word-level timestamps
- **PyDub**: Audio manipulation and processing library
- **Pydantic**: Data validation and settings management with type hints
- **Uvicorn**: ASGI server for running FastAPI applications

## Build System & Package Management

- **Poetry**: Dependency management and packaging tool
- **Python 3.12**: Required Python version (>=3.12, <3.13)

## Development Tools & Code Quality

- **Pre-commit**: Git hooks for code quality enforcement
- **Black**: Code formatter (line length: 88)
- **isort**: Import sorting (Black-compatible profile)
- **Ruff**: Fast Python linter and formatter
- **MyPy**: Static type checking
- **Bandit**: Security vulnerability scanner

## Testing Framework

- **pytest**: Testing framework with async support
- **pytest-asyncio**: Async test support
- **pytest-cov**: Coverage reporting
- **httpx**: HTTP client for API testing

## AI/ML Dependencies

- **torch**: PyTorch for deep learning models
- **openai-whisper**: Base Whisper implementation
- **numba**: JIT compilation for performance optimization

## Common Commands

### Development Setup
```bash
# Install dependencies
poetry install

# Install pre-commit hooks
pre-commit install

# Activate virtual environment
poetry shell
```

### Running the Application
```bash
# Development server with auto-reload
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production server
python app/main.py

# Using poetry
poetry run python app/main.py
```

### Code Quality & Testing
```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Format code
black .
isort .

# Lint code
ruff check .
ruff check . --fix

# Type checking
mypy .

# Security scan
bandit -r app/

# Run tests
pytest
pytest --cov=app
pytest -v --cov=app --cov-report=html
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:
- Model settings (DEFAULT_MODEL, DETECTOR_MODEL, DEVICE)
- Server configuration (HOST, PORT, WORKERS)
- Processing parameters (BATCH_SIZE, MIN_PROB)
- Logging levels and formats

## Code Style Standards

- **Line Length**: 88 characters (Black standard)
- **Import Organization**: isort with Black profile
- **Type Hints**: Required for all function signatures (MyPy strict mode)
- **Docstrings**: Google-style docstrings for classes and public methods
- **Error Handling**: Custom exception classes with structured error codes
- **Async/Await**: Prefer async patterns for I/O operations
