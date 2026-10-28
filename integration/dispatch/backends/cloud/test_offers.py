"""The offer catalog every cloud shares, and the card and architecture names it is asked by."""

from types import SimpleNamespace

import pytest

from mainboard import MissionError
from mainboard.costs import Offer
from mainboard.dispatch import arch, sm
from mainboard.dispatch.backends.cloud.offers import hunt
from mainboard.dispatch.jobs import OfferFilter, Pricing, capability, card
from mainboard.verbs.machines.rentals import Shopping


def test_a_card_reads_the_same_however_it_is_spelled() -> None:
    assert card("NVIDIA_GeForce_RTX_4090") == card("rtx 4090") == "RTX4090"


@pytest.mark.parametrize(
    ("spelled", "priced"),
    [
        pytest.param("on-demand", Pricing.ON_DEMAND, id="the-help-spelling"),
        pytest.param("on_demand", Pricing.ON_DEMAND, id="the-member-name"),
        pytest.param("spot", Pricing.SPOT, id="spot"),
    ],
)
def test_host_offers_takes_pricing_as_the_help_spells_it(spelled: str, priced: Pricing) -> None:
    assert Shopping.model_validate({"pricing": spelled}).wanted("H100").pricing is priced


def test_a_filter_admits_the_offers_it_asks_for_and_no_others() -> None:
    held = Offer(provider="x", gpu="H100", gpu_count=1, spot=False, rate_usd_hr=3.0)
    asked = OfferFilter(gpu_name="h100", gpus=1, pricing=Pricing.ON_DEMAND, max_usd_hr=3.5)
    assert asked.admits(held) and OfferFilter(arch=arch("sm_90+")).admits(held)
    refused = [
        OfferFilter(gpu_name="H200"),
        OfferFilter(gpus=8),
        OfferFilter(max_usd_hr=2.0),
        OfferFilter(pricing=Pricing.SPOT),
        OfferFilter(arch=arch("sm_100")),
    ]
    assert not any(each.admits(held) for each in refused)


def test_offers_list_every_cloud_and_mark_what_mb_rents(mb) -> None:
    ran = mb("host", "offers", "RTX 4090", "--limit", "40", timeout=300)
    assert ran.code == 0, ran.said
    rows = ran.out.splitlines()
    assert rows[0].startswith("provider") and any("vast" in row for row in rows[1:])


def test_one_catalog_failing_leaves_every_other_clouds_offers(monkeypatch) -> None:
    """A 502 from one cloud's API once ended `host offers` as a traceback."""
    gpuhunt = pytest.importorskip("gpuhunt")

    def answered(*, provider: list[str] | None = None, **_filters) -> list[SimpleNamespace]:
        if provider is None or provider == ["vultr"]:
            raise OSError("502 Server Error: Bad Gateway")
        return [SimpleNamespace(provider=provider[0], gpu_name="RTX4090", price=0.5)]

    loaded = SimpleNamespace(providers=[SimpleNamespace(NAME="vultr"), SimpleNamespace(NAME="x")])
    monkeypatch.setattr(gpuhunt, "query", answered)
    monkeypatch.setattr(gpuhunt, "default_catalog", lambda: loaded)
    found = {item.provider for item in hunt(OfferFilter(gpu_name="RTX 4090"))}
    assert "x" in found and "runpod" in found and "vultr" not in found
    assert hunt(OfferFilter(), providers=["vultr"]) == []


def test_an_architecture_matches_the_capability_its_kernels_load_on() -> None:
    rtx = arch("sm_120")
    assert {(each.low, each.high) for each in (rtx, arch("12.0"), arch("SM120"))} == {
        ((12, 0), (12, 0))
    }
    assert rtx.holds(capability("RTX PRO 4500")) and rtx.holds(capability("RTX 5090"))
    assert not rtx.holds(capability("B200")) and not rtx.holds(capability("GB10"))
    assert arch("blackwell-dc").holds(capability("B300"))  # sm_103, which gpuhunt lists as sm_100
    assert arch("sm_90+").holds(capability("B200")) and not arch("sm_90+").holds(
        capability("L40S")
    )
    assert sm(capability("RTX 4090")) == "sm_89" and capability("no such card") is None
    with pytest.raises(MissionError):
        arch("volta-ish")
