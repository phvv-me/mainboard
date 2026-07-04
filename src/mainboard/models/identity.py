import getpass
import os
import sys
from abc import ABC, abstractmethod

from .. import shell

if sys.platform != "win32":  # pragma: no cover - Windows has no POSIX group database
    import grp


class Identity(ABC):
    """Who is running and their group membership, resolved for the host OS.

    The concept is uniform across platforms — a login name, a primary group, and
    every group the user belongs to — while the source differs: the POSIX group
    database on Unix, `whoami` on Windows. `Identity.current()` returns the
    implementation for the running operating system.
    """

    @staticmethod
    def current() -> Identity:
        """The identity probe for the running operating system."""
        return WindowsIdentity() if sys.platform == "win32" else PosixIdentity()

    def user(self) -> str:
        """Login name of the current user, or "" when it cannot be determined."""
        try:
            return getpass.getuser()
        except KeyError, OSError:
            return ""

    @abstractmethod
    def primary_group(self) -> str:
        """Primary group name of the current user, or "" when it cannot be resolved."""

    @abstractmethod
    def all_groups(self) -> tuple[str, ...]:
        """Every group the current user belongs to."""


class PosixIdentity(Identity):
    """Identity resolved from the POSIX user and group database."""

    def primary_group(self) -> str:
        try:
            return grp.getgrgid(os.getgid()).gr_name
        except KeyError, OSError:
            return ""

    def all_groups(self) -> tuple[str, ...]:
        try:
            return tuple(grp.getgrgid(gid).gr_name for gid in os.getgroups())
        except KeyError, OSError:
            return ()


class WindowsIdentity(Identity):
    """Identity resolved from Windows `whoami`, which lists group membership.

    Windows has no single primary-group concept, so the first reported group
    stands in for it.
    """

    def primary_group(self) -> str:
        groups = self.all_groups()
        return groups[0] if groups else ""

    def all_groups(self) -> tuple[str, ...]:
        return shell.whoami_groups()
