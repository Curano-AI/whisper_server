class _DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        """Simplistic tokenization for tests."""
        return [ord(c) % 255 for c in text]


def get_tokenizer(*_args, **_kwargs) -> _DummyTokenizer:
    """Return a dummy tokenizer instance."""
    return _DummyTokenizer()
