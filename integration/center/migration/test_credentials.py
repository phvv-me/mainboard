"""`Credentials.sign_in` hands the destination this machine's gh login, or says there is none."""

import subprocess
from collections.abc import Mapping

import pytest

from mainboard.center.migration.steps.credentials import Credentials, Signed
from mainboard.core import Verdict

from ...fakes import Spawner, exiting, failing


class _Destination:
    """A destination whose gh takes every login, recording what it was handed."""

    def __init__(self) -> None:
        self.logins: list[str] = []

    def call(self, function: str, arguments: Mapping[str, str], shape: type[Signed]) -> Signed:
        self.logins.append(arguments["token"])
        return shape(signed=True)


def _signed_in(gh: Spawner, monkeypatch: pytest.MonkeyPatch) -> tuple[Verdict, list[str]]:
    """The verdict of signing in when `gh` answers as given, and the logins it handed on."""
    monkeypatch.setattr(subprocess, "run", gh)
    destination = _Destination()
    return Credentials("alias").sign_in(destination).verdict, destination.logins


def test_sign_in_hands_the_destination_the_token_gh_prints_without_its_newline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _signed_in(exiting(0, "tok\n"), monkeypatch) == (Verdict.PASS, ["tok"])


@pytest.mark.parametrize(
    "gh",
    [
        pytest.param(failing(FileNotFoundError("gh")), id="no-gh-on-the-path"),
        pytest.param(failing(subprocess.TimeoutExpired("gh", 30)), id="a-gh-past-its-timeout"),
        pytest.param(exiting(1, "tok\n"), id="a-gh-that-refuses-after-printing"),
    ],
)
def test_sign_in_carries_nothing_when_gh_has_no_login_to_give(
    gh: Spawner, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _signed_in(gh, monkeypatch) == (Verdict.WARN, [])
