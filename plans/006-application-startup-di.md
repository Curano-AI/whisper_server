# Plan 006 - Application startup and dependency injection

## Objective
Implement task 13 from `whisperx-fastapi-server` specs: "Create application startup and dependency injection".

## Proposed Steps
1. Add a new `app.dependencies` module providing `get_settings`, `get_model_manager`,
   `get_transcription_service` and `get_health_service` helpers that read objects
   from `request.app.state`.
2. Update `app/main.py` startup and shutdown events:
   - Instantiate `AppConfig` and store as `app.state.settings`.
   - Create `ModelManager`, `TranscriptionService`, and `HealthService` instances
     and store them in `app.state`.
   - Preload the default model via `ModelManager.load_model` and log
     configuration.
   - Perform a health check via `HealthService.get_health()` to validate
     dependencies.
   - On shutdown, unload all models using `ModelManager.clear()`.
3. Refactor API routers (`health.py`, `v1/models.py`, `v1/transcriptions.py`) to
   obtain services and configuration via the dependency helpers instead of module
   level globals.
4. Adapt existing tests to use `app.dependency_overrides` for injecting mocks and
   adjust settings through `app.state.settings`.
5. Add new tests in `tests/test_app_lifecycle.py` verifying that startup loads
   the default model and shutdown clears it.
6. Mark task 13 done in `tasks.md` with a changelog entry and update
   `test-summary.md` after running the suite.

## Risks / Open Questions
- Global `app` instance is reused across tests; ensuring clean state between
  tests relies on shutdown events correctly resetting model caches.
- Preloading the default model uses the stubbed `whisperx.load_model` during
  tests so it remains lightweight.

## Rollback Strategy
Revert modified files and remove `plans/006-application-startup-di.md`.
