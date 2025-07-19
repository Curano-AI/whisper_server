class _Cuda:
    def is_available(self) -> bool:
        """Pretend CUDA is unavailable for tests."""
        return False

    def empty_cache(self) -> None:
        """No-op cache clearing used in tests."""
        pass


cuda = _Cuda()
