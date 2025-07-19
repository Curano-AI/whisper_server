# Plan 004 - Implement health check and monitoring endpoints

## Objective
Implement task 10 from `whisperx-fastapi-server` specs: "Implement health check and monitoring endpoints".

## Proposed Steps
1. Add new `HealthService` in `app/services/health.py` providing method `get_health()` returning `HealthCheckResponse`.
   - Gather CPU, memory, and disk usage using `psutil`.
   - Detect GPU availability and memory usage via `torch.cuda`.
   - Include loaded model names via `ModelManager`.
2. Export `HealthService` from `app/services/__init__.py`.
3. Implement `/healthcheck` endpoint in `app/api/health.py` using a module-level `HealthService` instance.
   - Return `HealthCheckResponse` on success.
   - On unexpected failure, return `ErrorResponse` with status 503.
4. Write tests `tests/test_api_health.py` covering:
   - Successful health check response with mocked service data.
   - Error path when service raises `ResourceError`.
5. Update spec task list marking task 10 complete with changelog entry.
6. Run formatting, linters and full test suite and update `test-summary.md`.

## Risks / Open Questions
- `psutil` is not currently a dependency; adding it may require updates to `pyproject.toml` and lock file.
- GPU metrics may be unavailable on CPU-only systems; implementation should handle absence gracefully.

## Rollback Strategy
Revert the added service, endpoint, tests, dependency changes, and tasks file to restore previous state.
