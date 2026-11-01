"""`serve`, `serve status` and `serve stop` are the owner's: an agent is refused before any run."""

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("words", "named"),
    [
        pytest.param(("serve",), "serve", id="the-default-that-installs-the-unit"),
        pytest.param(("serve", "status"), "serve status", id="the-status"),
        pytest.param(("serve", "stop"), "serve stop", id="the-stop"),
        pytest.param(("serve", "--json"), "serve", id="the-default-with-its-only-options"),
    ],
)
def test_every_serve_verb_refuses_an_agent_and_touches_nothing(
    mb, workspace: Path, words: tuple[str, ...], named: str
) -> None:
    ran = mb(*words, env={"MB_AGENT": "1"})
    assert (ran.code, f"`mb {named}` is the owner's act" in ran.err) == (1, True)
    assert not (workspace / ".mb" / "serve").exists()


def test_the_serve_group_lists_status_and_stop_to_anyone(mb) -> None:
    ran = mb("serve", "--help", env={"MB_AGENT": "1"})
    assert ran.code == 0
    assert "status:" in ran.out and "stop:" in ran.out
