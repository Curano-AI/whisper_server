# Plan 001 - Implement model management endpoints

## Objective
Implement task 9 from `whisperx-fastapi-server` specs: "Implement model management endpoints".

## Proposed Steps
1. Create a module level `ModelManager` instance in `app/api/v1/models.py` for reuse.
2. Implement three endpoints using Pydantic models and `ModelManager`:
   - **GET /models/list** → return `LoadedModelsResponse` built from manager state.
   - **POST /models/load** → accept `ModelLoadRequest`, load model via manager, return `ModelLoadResponse` with metadata. Catch `ModelLoadError` and return `ErrorResponse` with status 500.
   - **POST /models/unload** → accept `ModelUnloadRequest`, unload model via manager, return `ModelUnloadResponse`.
3. Write tests in `tests/test_api_models.py` covering success paths and an error case for `/models/load`.
4. Update `.kiro/specs/whisperx-fastapi-server/tasks.md` marking task 9 done with changelog entry.
5. Run linters and tests per AGENTS.md.

## Risks / Open Questions
- The service currently lacks dependency injection; using a module-level manager may limit test isolation but simplest for now.
- Memory usage information is not tracked; will return `None` for that field.

## Rollback Strategy
Revert the commit or restore previous `app/api/v1/models.py`, tests, and tasks file using git.
