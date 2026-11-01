"""The host's `ld.so.cache` as `ldconfig -p` lists it, empty whenever the host cannot say."""

from pathlib import Path

import pytest

from mainboard.libs.loader import system_cache

_LISTING = r"""
echo "3 libs found in cache \`/etc/ld.so.cache'"
printf '\tlibc.so.6 (libc6,x86-64) => /lib/libc.so.6\n'
printf '\tlibc.so.6 (libc6) => /lib32/libc.so.6\n'
printf '\tlibm.so.6 (libc6,x86-64) => /lib/libm.so.6\n'
"""
_STALLED = r"""
printf '\tlibc.so.6 (libc6,x86-64) => /lib/libc.so.6\n'
exec sleep 30
"""


@pytest.mark.parametrize(
    ("script", "cache"),
    [
        pytest.param(
            _LISTING,
            {"libc.so.6": Path("/lib/libc.so.6"), "libm.so.6": Path("/lib/libm.so.6")},
            id="the-first-entry-of-a-soname-wins",
        ),
        pytest.param(_STALLED, {}, id="a-listing-cut-short-by-the-deadline"),
    ],
)
def test_the_cache_is_what_ldconfig_lists_and_empty_when_it_stalls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script: str,
    cache: dict[str, Path],
) -> None:
    ldconfig = tmp_path / "ldconfig"
    ldconfig.write_text("#!/bin/sh\n" + script)
    ldconfig.chmod(0o755)
    monkeypatch.setattr("shutil.which", lambda _: str(ldconfig))
    monkeypatch.setattr("mainboard.libs.loader.ASK_SECONDS", 0.5)

    assert system_cache() == cache


def test_the_cache_is_empty_where_ldconfig_cannot_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: str(tmp_path / "absent"))

    assert system_cache() == {}
