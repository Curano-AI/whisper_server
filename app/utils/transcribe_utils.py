"""Utility functions ported from transcribe.py for WhisperX FastAPI server.

This module contains the core utility functions from the original transcribe.py script,
including timestamp formatting, text cleaning, audio processing, and token suppression.
"""

import os
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import cast

from pydub import AudioSegment
from whisper.tokenizer import get_tokenizer


def ts(sec: float) -> str:
    """Convert seconds to SRT timestamp format.

    Converts a floating-point seconds value to the SRT subtitle format:
    HH:MM:SS,mmm (hours:minutes:seconds,milliseconds)

    Args:
        sec: Time in seconds (can be float)

    Returns:
        Formatted timestamp string in SRT format

    Example:
        >>> ts(65.123)
        '00:01:05,123'
        >>> ts(3661.456)
        '01:01:01,456'
    """
    td = timedelta(seconds=sec)
    total_seconds = int(td.total_seconds())

    # Calculate hours, minutes, seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    # Calculate milliseconds from the fractional part
    milliseconds = round((sec - int(sec)) * 1000)

    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def clean(txt: str, suppress_phrases: list[str] | None = None) -> str:
    """Remove suppressed phrases from transcription text.

    Removes unwanted phrases from the transcription text and normalizes whitespace.
    Uses the default suppress phrases from transcribe.py if none provided.

    Args:
        txt: Input text to clean
        suppress_phrases: List of phrases to remove (optional)

    Returns:
        Cleaned text with suppressed phrases removed and normalized whitespace

    Example:
        >>> clean("Hello дима торжок world")
        'Hello world'
    """
    if suppress_phrases is None:
        suppress_phrases = [
            "дима торжок",
            "dima torzok",
            "dima torzhok",
            "субтитры подогнал",
        ]

    for phrase in suppress_phrases:
        txt = txt.replace(phrase, "")

    return " ".join(txt.split())


def export_chunk(audio: AudioSegment, start_ms: int, dur: int = 10_000) -> str:
    """Export audio chunk to temporary WAV file.

    Creates a temporary WAV file containing a chunk of audio starting at the
    specified position with the given duration.

    Args:
        audio: AudioSegment object to extract chunk from
        start_ms: Start position in milliseconds
        dur: Duration of chunk in milliseconds (default: 10 seconds)

    Returns:
        Path to the temporary WAV file

    Note:
        The caller is responsible for cleaning up the temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        chunk = cast("AudioSegment", audio[start_ms : start_ms + dur])
        chunk.export(tmp.name, format="wav")
        return tmp.name


def get_suppress_tokens(suppress_phrases: list[str] | None = None) -> list[int]:
    """Generate suppress tokens from phrases using whisper tokenizer.

    Converts text phrases to token IDs that should be suppressed during transcription.
    Uses the default suppress phrases from transcribe.py if none provided.

    Args:
        suppress_phrases: List of phrases to convert to tokens (optional)

    Returns:
        Sorted list of token IDs to suppress

    Example:
        >>> tokens = get_suppress_tokens(["hello world"])
        >>> isinstance(tokens, list)
        True
        >>> all(isinstance(t, int) for t in tokens)
        True
    """
    if suppress_phrases is None:
        suppress_phrases = [
            "дима торжок",
            "dima torzok",
            "dima torzhok",
            "субтитры подогнал",
        ]

    tok = get_tokenizer(multilingual=True)
    suppress_tokens: set[int] = set()

    for phrase in suppress_phrases:
        tokens = tok.encode(phrase)
        suppress_tokens.update(tokens)

    return sorted(suppress_tokens)


def cleanup_temp_files(file_paths: list[str]) -> None:
    """Clean up temporary files.

    Safely removes temporary files, ignoring any that don't exist.

    Args:
        file_paths: List of file paths to remove
    """
    for file_path in file_paths:
        try:
            if Path(file_path).exists():
                os.unlink(file_path)
        except OSError:
            # Ignore errors when cleaning up temp files
            pass


# Default suppress phrases from transcribe.py
DEFAULT_SUPPRESS_PHRASES = [
    "дима торжок",
    "dima torzok",
    "dima torzhok",
    "субтитры подогнал",
]
