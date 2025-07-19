# Plan 003 - Implement POST /v1/audio/transcriptions endpoint

## Objective
Implement task 8 from `whisperx-fastapi-server` specs: "Implement POST /v1/audio/transcriptions endpoint".

## Proposed Steps
1. Add `TranscriptionService` instance in `app/api/v1/transcriptions.py` and implement endpoint logic.
   - Save uploaded file to a temporary location.
   - Enforce `AppConfig.max_file_size` limit and return 400 if exceeded.
   - Call `service.transcribe()` with the temp file path and request model.
   - Format plain text/SRT/VTT responses using `fastapi.Response` with correct media type.
   - Handle `AudioProcessingError` and `TranscriptionError` returning OpenAI compatible errors.
   - Always clean up temporary file.
2. Update `__all__` exports if needed (none).
3. Create tests `tests/test_api_transcriptions.py` covering:
   - Successful JSON response using mocked service.
   - Successful text response (`response_format="text"`).
   - File size limit exceeded returning error code `file_too_large`.
4. Update `.kiro/specs/whisperx-fastapi-server/tasks.md` marking task 8 done and add changelog bullet.
5. Run formatting, linting and full test suite and update `test-summary.md`.

## Risks / Open Questions
- Temporary file cleanup must be ensured even on errors.
- Media type mapping for formats might require adjustment later.

## Rollback Strategy
Revert changes to endpoint, tests and spec task file to restore previous state.
