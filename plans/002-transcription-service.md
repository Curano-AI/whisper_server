# Plan 002 - Implement core TranscriptionService

## Objective
Implement task 7 from `whisperx-fastapi-server` specs: "Implement core TranscriptionService".

## Proposed Steps
1. Create new `app/services/transcription.py` implementing `TranscriptionService`.
   - Use `AudioProcessor`, `LanguageDetector` and `ModelManager`.
   - Provide method `transcribe(file_path: str, req: TranscriptionRequest) -> TranscriptionOutput`.
   - Build ASR options exactly from `AppConfig.get_asr_options()` and request overrides
     (beam_size, temperature, suppress_tokens, etc.).
   - Apply VAD options from settings with request overrides.
   - Load model through `ModelManager`, run model `transcribe` on file.
   - Format response for `json`, `verbose_json`, `text`, `srt`, `vtt` using `ts` and `clean` utils.
2. Update `app/services/__init__.py` to export the new service.
3. Add unit tests in `tests/test_transcription_service.py` mocking underlying
   services to verify orchestration and response formatting.
4. Mark task 7 done with changelog bullet in spec and update test summary after running tests.

## Risks / Open Questions
- Simplified response formatting may not cover every WhisperX field but will follow
  existing response models.
- Integration tests rely heavily on mocks since real model inference is heavy.

## Rollback Strategy
Revert added files and tasks update to restore previous state.
