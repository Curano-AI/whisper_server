# WhisperX FastAPI Server

OpenAI-compatible audio transcription service using WhisperX with intelligent language detection.

## Project Structure

```
app/
├── main.py                 # FastAPI app initialization
├── api/
│   ├── v1/
│   │   ├── transcriptions.py  # /v1/audio/transcriptions
│   │   └── models.py          # Model management endpoints
│   └── health.py           # Health check endpoints
├── core/
│   ├── config.py          # Application configuration
│   ├── exceptions.py      # Custom exceptions
│   └── logging.py         # Logging configuration
├── services/              # Business logic (to be implemented)
├── models/               # Pydantic models (to be implemented)
└── utils/                # Utility functions (to be implemented)
```

## Configuration

The server uses Pydantic settings with defaults from the original transcribe.py script:

- **Model Configuration**: large-v3 (default), small (detector)
- **Audio Processing**: Silence trimming (300ms, -35dB), 10s chunks for language detection
- **Language Detection**: 0.6 confidence threshold, vote counting with confidence tie-breaking
- **ASR Options**: beam_size=2, no condition on previous text, temperature=0.0
- **VAD Options**: 500ms min silence, 200ms speech padding
- **Token Suppression**: Configured phrases from original script

## Environment Variables

Copy `.env.example` to `.env` and configure as needed:

```bash
HOST=0.0.0.0
PORT=8000
DEFAULT_MODEL=large-v3
DEVICE=cuda
LOG_LEVEL=INFO
```

## Running the Server

```bash
# Install dependencies
poetry install

# Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or use the main module
python app/main.py
```

## API Documentation

Once running, visit:

- OpenAPI docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
