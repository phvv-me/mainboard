from collections.abc import Callable

import pytest


@pytest.fixture
def which(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """A patcher for `shutil.which` that reports only the named binaries as installed."""

    def patch(*installed: str) -> None:
        monkeypatch.setattr(
            "shutil.which",
            lambda binary: f"/usr/bin/{binary}" if binary in installed else None,
        )

    return patch
