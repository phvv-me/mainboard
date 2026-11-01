"""What `host offers` ranks and shows, from recorded offers and landings; no cloud is asked."""

from pathlib import Path

import pytest

from mainboard.dispatch.backends import ProviderBackend
from mainboard.policy.grants import Grants
from mainboard.verbs.machines.rentals import RentalVerbs

from ...policy.builders import Fault, card, spoiled
from .readers import table


def _offers(capsys: pytest.CaptureFixture[str]) -> list[dict[str, str]]:
    """The rows `host offers` prints for an RTX 4090."""
    RentalVerbs().offers("RTX 4090")
    return table(capsys.readouterr().out)


def test_offers_without_a_record_keep_their_price_order_and_show_no_measurement(
    bare, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = _offers(capsys)
    assert [row["provider"] for row in rows] == ["aws", "vastai", "runpod", "lambdalabs"]
    assert not {"useful_usd_hr", "landed", "ready_s"} & set(rows[0])


def test_offers_rank_by_the_record_of_each_provider_at_equal_prices(
    measured, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = {row["provider"]: row for row in _offers(capsys)}
    assert list(rows) == ["aws", "runpod", "vastai", "lambdalabs"]
    assert (rows["runpod"]["landed"], rows["runpod"]["ready_s"]) == ("0.857", "40.0")
    assert rows["vastai"]["useful_usd_hr"] > rows["runpod"]["useful_usd_hr"]
    assert rows["aws"]["useful_usd_hr"] == "" and rows["lambdalabs"]["landed"] == ""


def test_offers_show_the_cap_on_the_bill_of_each_provider_mb_rents(
    bare, capsys: pytest.CaptureFixture[str]
) -> None:
    caps = {row["provider"]: row["cap"] for row in _offers(capsys)}
    assert caps == {
        "aws": "",
        "vastai": "prepaid",
        "runpod": "prepaid",
        "lambdalabs": "card, no limit declared",
    }


@pytest.mark.parametrize(
    ("kind", "grants", "balance", "shown"),
    [
        pytest.param("lambda", Grants(), None, "card, no limit declared", id="a-card-nobody-set"),
        pytest.param(
            "lambda",
            Grants(cards=(card("lambda", usd=100.0),)),
            None,
            "card $100.00",
            id="a-card-with-its-limit",
        ),
        pytest.param("vast", Grants(), None, "prepaid", id="a-prepaid-balance-nobody-read"),
        pytest.param("vast", Grants(), 12.3, "prepaid $12.30", id="a-prepaid-balance-as-reported"),
        pytest.param(
            "shadeform", Grants(), 12.3, "threshold", id="a-threshold-each-rental-carries"
        ),
        pytest.param("local", Grants(), None, "", id="a-host-that-is-no-provider"),
    ],
)
def test_a_cap_is_shown_as_the_bound_and_what_is_left_of_it(
    kind: str, grants: Grants, balance: float | None, shown: str
) -> None:
    assert RentalVerbs.capped(kind, grants=grants, balance_usd=balance) == shown


@pytest.mark.parametrize(
    "fault",
    [
        pytest.param("no-text", id="a-byte-that-is-no-utf8"),
        pytest.param("unreadable", id="mode-000"),
        pytest.param("a-pipe", id="a-fifo-nobody-writes"),
    ],
)
def test_a_grants_file_mb_cannot_read_leaves_the_listings_a_warning_and_no_card_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fault: Fault,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    spoiled(tmp_path, fault=fault)
    grants = RentalVerbs.granted(tmp_path)
    assert (grants, "the grants file was refused" in capsys.readouterr().err) == (Grants(), True)


def test_every_provider_mb_can_rent_through_shows_a_cap() -> None:
    shown = {kind: RentalVerbs.capped(kind, grants=Grants()) for kind in ProviderBackend.names()}
    assert not [kind for kind, cap in shown.items() if not cap]
