"""The card lease: one per physical card for the whole host, taken and given back atomically."""

from pathlib import Path

import pytest

from mainboard.trials.lease import Busy, CardLease

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
