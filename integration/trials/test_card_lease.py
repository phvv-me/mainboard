"""The card lease: one per physical card for the whole host, taken and given back atomically."""

import os
from pathlib import Path

import pytest

from mainboard.trials import Busy, CardLease

_CARD = "GPU-7d2f0c3e-1111-2222-3333-444455556666"


def test_a_live_lease_refuses_every_other_session_on_its_card(tmp_path: Path) -> None:
    held = CardLease.acquire(_CARD, home=tmp_path)

    with pytest.raises(Busy, match="refusing to measure beside it"):
        CardLease.acquire(_CARD, home=tmp_path)
    other = CardLease.acquire("GPU-another-card", home=tmp_path)

    held.release()
    other.release()
    CardLease.acquire(_CARD, home=tmp_path).release()
    assert not list(tmp_path.iterdir()), "no draft, stale or released file is left behind"


def test_a_dead_holders_lease_is_reclaimed_and_its_late_release_spares_the_successor(
    tmp_path: Path,
) -> None:
    first = CardLease.acquire(_CARD, home=tmp_path)
    first.path.write_text(f"999999999 {0.0} {first.token}", encoding="utf-8")

    successor = CardLease.acquire(_CARD, home=tmp_path)
    first.release()

    assert successor.path.read_text(encoding="utf-8").split()[2] == successor.token
    with pytest.raises(Busy):
        CardLease.acquire(_CARD, home=tmp_path)


def test_a_refusal_names_the_live_holders_pid_and_how_long_it_has_held(tmp_path: Path) -> None:
    held = CardLease.acquire(_CARD, home=tmp_path)

    with pytest.raises(Busy) as refused:
        CardLease.acquire(_CARD, home=tmp_path)

    assert refused.value.pid == os.getpid()
    assert 0 <= refused.value.age < 60
    held.release()


def test_a_live_lease_past_its_ttl_is_reclaimed_and_its_late_release_spares_the_successor(
    tmp_path: Path,
) -> None:
    outrun = CardLease.acquire(_CARD, home=tmp_path, ttl=0.0)

    successor = CardLease.acquire(_CARD, home=tmp_path, ttl=0.0)
    outrun.release()

    assert successor.path.exists()
    successor.release()
    assert not list(tmp_path.iterdir()), "only the successor's marker stood, and it is gone"
