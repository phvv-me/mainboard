from __future__ import annotations

import sys

import pytest

from mainboard.models import identity as identity_mod
from mainboard.models.identity import Identity, PosixIdentity, WindowsIdentity

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX group database is Unix-only"
)


def test_user_returns_login_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """`user` returns the OS login name (the lookup is platform-neutral)."""
    monkeypatch.setattr(identity_mod.getpass, "getuser", lambda: "alice")
    assert WindowsIdentity().user() == "alice"


def test_user_tolerates_lookup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed username lookup yields "" instead of raising."""

    def boom() -> str:
        raise OSError("no login name")

    monkeypatch.setattr(identity_mod.getpass, "getuser", boom)
    assert WindowsIdentity().user() == ""


@posix_only
def test_posix_groups_resolve_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """`PosixIdentity` resolves gids to group names from the group database."""
    names = {10: "research", 20: "gpu"}
    monkeypatch.setattr(identity_mod.os, "getgid", lambda: 10)
    monkeypatch.setattr(identity_mod.os, "getgroups", lambda: [10, 20])
    monkeypatch.setattr(
        identity_mod.grp, "getgrgid", lambda gid: type("G", (), {"gr_name": names[gid]})()
    )
    posix = PosixIdentity()
    assert posix.primary_group() == "research"
    assert posix.all_groups() == ("research", "gpu")


@posix_only
def test_posix_groups_tolerate_lookup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed group lookup yields empty defaults instead of raising."""

    def boom(_gid: int) -> object:
        raise KeyError("no such gid")

    monkeypatch.setattr(identity_mod.grp, "getgrgid", boom)
    posix = PosixIdentity()
    assert posix.primary_group() == ""
    assert posix.all_groups() == ()


def test_windows_groups_from_whoami(monkeypatch: pytest.MonkeyPatch) -> None:
    """`WindowsIdentity` reports the groups `whoami` lists, primary first."""
    monkeypatch.setattr(
        identity_mod.shell, "whoami_groups", lambda: ("Everyone", "BUILTIN\\Users")
    )
    win = WindowsIdentity()
    assert win.all_groups() == ("Everyone", "BUILTIN\\Users")
    assert win.primary_group() == "Everyone"


def test_windows_primary_group_empty_without_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no groups reported, the Windows primary group is "" rather than an error."""
    monkeypatch.setattr(identity_mod.shell, "whoami_groups", lambda: ())
    assert WindowsIdentity().primary_group() == ""


def test_current_selects_posix_on_unix(monkeypatch: pytest.MonkeyPatch) -> None:
    """`current` returns the POSIX implementation on a Unix host."""
    monkeypatch.setattr(identity_mod.sys, "platform", "linux")
    assert isinstance(Identity.current(), PosixIdentity)


def test_current_selects_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """`current` returns the Windows implementation on a Windows host."""
    monkeypatch.setattr(identity_mod.sys, "platform", "win32")
    assert isinstance(Identity.current(), WindowsIdentity)
