# Plan 005-error-handling-middleware

## Objective
Implement comprehensive error handling middleware and logging according to task 11 in `whisperx-fastapi-server` spec.

## Proposed Steps
1. Create `app/core/error_handlers.py` with `register_exception_handlers(app)` registering handlers for `WhisperXAPIException` and generic `Exception` returning `ErrorResponse`.
2. Create `app/core/middleware.py` with `LoggingMiddleware` logging request start/end and duration.
3. Update `app/main.py` to add the middleware and call `register_exception_handlers`.
4. Refactor API endpoints to raise custom exceptions instead of returning `JSONResponse`.
   - `/healthcheck` remove try/except for `ResourceError`.
   - `/v1/audio/transcriptions` raise `ValidationError` for file too large and let `AudioProcessingError`/`TranscriptionError` bubble up.
   - `/models/load` and `/models/unload` raise `ModelLoadError` when necessary.
5. Write new tests in `tests/test_error_handlers.py` verifying error handler responses when exceptions are raised and for generic failures.
6. Update existing tests if needed to align with refactored endpoints.
7. Run formatting, linting and tests. Update `tasks.md` and `test-summary.md`.

## Risks / Open Questions
- New middleware might change response details; ensure existing tests still pass.
- Generic exception handler should not expose internal errors; return generic message.

## Rollback Strategy
Revert the commit or remove middleware registration and restore previous endpoint logic.
