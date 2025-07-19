# Project Structureanization

## Directory Layout

```
app/                          # Main application package
├── __init__.py              # Package initialization
├── main.py                  # FastAPI app initialization and startup
├── api/                     # API route handlers
│   ├── __init__.py
│   ├── health.py           # Health check endpoints
│   └── v1/                 # API version 1
│       ├── transcriptions.py  # /v1/audio/transcriptions
│       └── models.py          # Model management endpoints
├── core/                    # Core application components
│   ├── __init__.py
│   ├── config.py           # Pydantic settings and configuration
│   ├── exceptions.py       # Custom exception classes
│   └── logging.py          # Logging configuration
├── models/                  # Pydantic data models
│   ├── __init__.py
│   ├── requests.py         # API request models
│   └── responses.py        # API response models
├── services/                # Business logic layer
│   ├── __init__.py
│   ├── audio_processor.py  # Audio processing service
│   └── language_detector.py # Language detection service
└── utils/                   # Utility functions
    ├── __init__.py
    └── transcribe_utils.py  # Transcription utilities

tests/                       # Test suite
├── test_audio_processor.py
├── test_language_detector.py
├── test_models.py
└── test_transcribe_utils.py

scripts/                     # Utility scripts
.kiro/                       # Kiro IDE configuration
├── steering/               # AI assistant steering rules
└── specs/                  # Feature specifications
```

## Architecture Patterns

### Layered Architecture
- **API Layer** (`app/api/`): FastAPI route handlers, request/response handling
- **Service Layer** (`app/services/`): Business logic, audio processing, ML operations
- **Model Layer** (`app/models/`): Pydantic models for data validation
- **Core Layer** (`app/core/`): Configuration, exceptions, logging, shared utilities

### Dependency Injection
- Configuration managed through `app/core/config.py` with Pydantic settings
- Services instantiated with dependency injection pattern
- Environment-based configuration with `.env` file support

### Error Handling Strategy
- Custom exception hierarchy in `app/core/exceptions.py`
- Structured error responses with error codes and parameters
- OpenAI-compatible error format for API responses

## File Naming Conventions

### Python Files
- **Snake case**: `audio_processor.py`, `language_detector.py`
- **Descriptive names**: Files should clearly indicate their purpose
- **Test files**: Prefix with `test_` matching the module name

### Classes and Functions
- **Classes**: PascalCase (`AudioProcessor`, `LanguageDetector`)
- **Functions/Methods**: snake_case (`process_audio`, `detect_language`)
- **Constants**: UPPER_SNAKE_CASE (`SILENCE_THRESHOLD`, `CHUNK_DURATION`)

## Import Organization

### Import Order (isort + Black compatible)
1. Standard library imports
2. Third-party library imports
3. Local application imports (using `from app.` prefix)

### Example Import Structure
```python
import logging
from pathlib import Path
from typing import Any

import whisperx
from fastapi import FastAPI
from pydub import AudioSegment

from app.core.config import get_settings
from app.core.exceptions import AudioProcessingError
from app.services.audio_processor import AudioProcessor
```

## Configuration Management

### Settings Hierarchy
1. **Default values** in `AppConfig` class
2. **Environment variables** (`.env` file)
3. **Runtime overrides** via dependency injection

### Environment Files
- `.env.example`: Template with all available settings
- `.env`: Local development configuration (gitignored)
- Production: Environment variables or container configuration

## Testing Structure

### Test Organization
- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Fixtures**: Shared test data and mock objects
- **Markers**: `@pytest.mark.slow`, `@pytest.mark.integration`

### Test Naming
- Test files: `test_{module_name}.py`
- Test classes: `Test{ClassName}`
- Test methods: `test_{functionality}_{condition}`

## Documentation Standards

### Docstring Format (Google Style)
```python
def process_audio(self, file_path: str) -> AudioSegment:
    """Process audio file for transcription.

    Args:
        file_path: Path to the audio file to process

    Returns:
        Processed AudioSegment ready for transcription

    Raises:
        AudioProcessingError: If audio processing fails
    """
```

### Type Hints
- **Required**: All public methods and functions must have type hints
- **Union types**: Use `str | None` (Python 3.10+ syntax)
- **Generic types**: Use `list[str]`, `dict[str, Any]`

## Service Layer Patterns

### Context Managers
- Services that manage resources should implement context manager protocol
- Automatic cleanup of temporary files and model memory

### Error Propagation
- Services raise domain-specific exceptions
- API layer catches and converts to HTTP responses
- Structured error codes for client handling

### Async Patterns
- Use async/await for I/O operations
- FastAPI endpoints should be async when calling async services
- Blocking operations (ML inference) run in thread pools
