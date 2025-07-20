# Plan 007 - production deployment configuration

## Objective
Implement task 14 from `whisperx-fastapi-server` specs: "Add production deployment configuration".

## Proposed Steps
1. Create `Dockerfile` for building the FastAPI server using Python 3.12 image, installing dependencies with Poetry and running `uvicorn app.main:app`.
2. Extend `.env.example` with full set of configurable environment variables used by `AppConfig`.
3. Add `SecurityHeadersMiddleware` in new module `app.core.security` and register it in `app.main` to set standard security headers.
4. Update `README.md` with a "Deployment" section explaining Docker usage and environment configuration.
5. Write `tests/test_deployment.py` verifying environment variables override defaults and security headers are included in responses.
6. Mark task 14 done and update test summary after running the suite.

## Risks / Open Questions
- Dockerfile may need additional system packages (e.g. ffmpeg) for WhisperX; slim base with ffmpeg should suffice.
- Existing caching of `get_settings` requires clearing in tests when modifying env vars.

## Rollback Strategy
Remove the new files and revert modifications to restore previous state.
