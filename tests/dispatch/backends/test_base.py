import os
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from threading import Event, Thread
from time import sleep
from types import SimpleNamespace
from urllib.request import Request

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.dispatch.backends import (
    Account,
    Capability,
    Credentials,
    Delivery,
    HpcAiBackend,
    LogSource,
    Market,
    ModalBackend,
    ProviderBackend,
    Standing,
    VastBackend,
    base,
    http_transport,
    route,
)
from mainboard.dispatch.backends import api_key as hpc_ai_key
from mainboard.dispatch.backends.base import image_cuda
from mainboard.dispatch.backends.modal import declared_credit
from mainboard.dispatch.backends.vast import api_key as vast_key
from mainboard.dispatch.vocabulary import Resources
from mainboard.manifest import Container, HostProfile

from .support import BareBackend, FakeTransport, hpc_ai_backend, plan, vast_backend

# Money as a provider really quotes it, from a free rental up to a balance nobody has.
_MONEY = st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False)


class Blocking:
    """A `Project` stand-in holding the merge open until the test lets it finish."""

    def __init__(self, root: Path, reading: Event, may_finish: Event) -> None:
        self.root = root
        self.reading = reading
        self.may_finish = may_finish

    def find_root(self, start: Path) -> Path:
        del start
        self.reading.set()
        self.may_finish.wait(timeout=5)
        return self.root


def test_the_workspace_env_defines_only_what_the_environment_lacks_and_is_read_once(
    unsealed: None, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file read as data: comments, blanks and junk skipped, quotes off, the environment wins.

    The second load proves the merge happens once, so a rewritten file never overwrites a value a
    provider already holds.
    """
    monkeypatch.chdir(workspace)
    (workspace / ".env").write_text(
        """# provider keys

export VAST_API_KEY=from-the-file
HPCAI_API_KEY='already exported'
MODAL_CREDIT_USD = 30
a line that declares nothing
=headless
"""
    )
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("MODAL_CREDIT_USD", raising=False)
    monkeypatch.setenv("HPCAI_API_KEY", "kept")
    assert Credentials().load() == ("VAST_API_KEY", "MODAL_CREDIT_USD")
    assert os.environ["VAST_API_KEY"] == "from-the-file"
    assert os.environ["MODAL_CREDIT_USD"] == "30"
    assert os.environ["HPCAI_API_KEY"] == "kept"
    (workspace / ".env").write_text("VAST_API_KEY=second\n")
    assert Credentials().load() == ()
    assert os.environ["VAST_API_KEY"] == "from-the-file"


def test_neither_a_workspace_without_an_env_file_nor_a_machine_outside_one_defines_anything(
    unsealed: None, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both are one plain empty answer rather than the refusal `find_root` raises."""
    monkeypatch.chdir(workspace)
    assert Credentials().load() == ()
    Credentials().loaded = False
    monkeypatch.chdir(workspace.parent)
    assert Credentials().load() == ()


def test_a_second_backend_asking_at_once_waits_for_the_whole_merge(
    unsealed: None, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compute survey probes every provider at once, so a flag flipped before the read would
    report a paid account unkeyed for no reason but timing.
    """
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    (workspace / ".env").write_text("VAST_API_KEY=one\n")
    credentials = Credentials()
    reading, may_finish = Event(), Event()
    monkeypatch.setattr(credentials, "project", Blocking(workspace, reading, may_finish))
    seen: list[str | None] = []

    def second() -> None:
        credentials.load()
        seen.append(os.environ.get("VAST_API_KEY"))

    first, other = Thread(target=credentials.load), Thread(target=second)
    first.start()
    reading.wait(timeout=5)
    other.start()
    other.join(timeout=0.03)
    assert other.is_alive()  # waiting on the merge rather than reporting an empty environment
    may_finish.set()
    first.join(timeout=5)
    other.join(timeout=5)
    assert seen == ["one"]


@pytest.mark.parametrize(
    ("reader", "variable", "declared", "expected"),
    [
        pytest.param(vast_key, "VAST_API_KEY", "from-the-env", "from-the-env", id="vast"),
        pytest.param(hpc_ai_key, "HPCAI_API_KEY", "from-the-env", "from-the-env", id="hpc-ai"),
        pytest.param(declared_credit, "MODAL_CREDIT_USD", "30", 30.0, id="modal"),
    ],
)
def test_a_providers_own_reader_finds_what_only_the_workspace_env_declares(
    reader: Callable[[], str | float],
    variable: str,
    declared: str,
    expected: str | float,
    unsealed: None,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each refusal tells someone to write that file, so each reader reads it."""
    monkeypatch.chdir(workspace)
    monkeypatch.delenv(variable, raising=False)
    (workspace / ".env").write_text(f"{variable}={declared}\n")
    assert reader() == expected


@pytest.mark.parametrize(
    ("kind", "path"),
    [
        pytest.param("ssh", "ssh-family", id="ssh"),
        pytest.param("pbs", "ssh-family", id="pbs"),
        pytest.param("slurm", "ssh-family", id="slurm"),
        pytest.param("local", "ssh-family", id="local"),
        pytest.param(HostProfile().kind, "ssh-family", id="an-unprobed-host-left-on-auto"),
        pytest.param("modal", ModalBackend, id="modal"),
        pytest.param("hpc-ai", HpcAiBackend, id="hpc-ai"),
        pytest.param("vast", VastBackend, id="vast"),
    ],
)
def test_route_sends_each_kind_down_the_path_that_can_run_it(
    kind: str, path: str | type[ProviderBackend]
) -> None:
    assert route(kind) == path


def test_route_raises_a_mission_error_naming_known_kinds_for_an_unregistered_kind() -> None:
    with pytest.raises(MissionError) as excinfo:
        route("ec2")
    assert "ec2" in str(excinfo.value)
    assert "modal" in str(excinfo.value)
    assert "hpc-ai" in str(excinfo.value)


def test_every_backend_carries_the_job_lifecycle_and_nothing_it_cannot_honor() -> None:
    """The capability map as a table, the lifecycle root carrying none of the contracts."""
    contracts = (Account, Delivery, LogSource, Market)
    assert all(issubclass(contract, Capability) for contract in contracts)
    assert not any(issubclass(ProviderBackend, contract) for contract in contracts)
    carried = {
        "modal": ModalBackend(),
        "hpc-ai": hpc_ai_backend(transport=FakeTransport()),
        "vast": vast_backend(),
        "bare": BareBackend(),
    }
    assert {
        name: sorted(contract.__name__ for contract in contracts if isinstance(backend, contract))
        for name, backend in carried.items()
    } == {
        "modal": ["Account", "LogSource"],
        "hpc-ai": ["Account"],
        "vast": ["Account", "LogSource", "Market"],
        "bare": [],
    }


@pytest.mark.parametrize(
    ("capability", "line"),
    [
        pytest.param(
            LogSource,
            "bare backend keeps no logs; read bare-1.log on the box instead",
            id="a-gap-the-backend-wrote-its-own-advice-for",
        ),
        pytest.param(
            Delivery,
            "the bare backend does not implement Delivery",
            id="a-gap-it-never-described",
        ),
    ],
)
def test_a_refusal_carries_the_backends_own_advice_or_a_plain_statement_of_the_gap(
    capability: type[Capability], line: str
) -> None:
    assert BareBackend().refusal(capability, handle="bare-1") == line


def test_the_audited_url_open_is_what_every_rest_backend_takes_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One seam reaches urllib, under the package's own deadline."""
    assert VastBackend().transport is http_transport
    assert HpcAiBackend().transport is http_transport
    assert VastBackend().sleeper is sleep
    seen: list[tuple[Request, float]] = []

    def fake_urlopen(request: Request, *, timeout: float) -> SimpleNamespace:
        seen.append((request, timeout))
        return SimpleNamespace(status=200, read=lambda: b"{}")

    monkeypatch.setattr(base, "urlopen", fake_urlopen)
    request = Request("https://example.test/probe")
    assert http_transport(request).read() == b"{}"
    seen_request, deadline = seen[0]
    assert seen_request is request
    assert 0 < deadline < 60


# Ten examples, since the rule is one `round` call and a validated model is this file's costliest
# build under coverage; the two live figures the rule was written for always run.
@settings(max_examples=10)
@given(figure=_MONEY)
@example(figure=99.99680725539)
@example(figure=0.285925925926)
def test_standing_quotes_money_at_the_precision_money_has(figure: float) -> None:
    """Four places, so two real hourly rates never read alike, and an absent figure stays None."""
    priced = Standing(keyed=True, credit_usd=figure, usd_hr=None)
    assert priced.credit_usd == pytest.approx(figure, abs=1e-4)
    assert priced.credit_usd == round(priced.credit_usd, 4)
    assert priced.usd_hr is None


@given(cap=_MONEY)
@example(cap=0.0)
@example(cap=5.0)
def test_admit_refuses_a_submission_nobody_capped(cap: float) -> None:
    """An uncapped submit is refused before any network call."""
    refuses = pytest.raises(MissionError, match="max-usd") if not cap else nullcontext()
    with refuses:
        BareBackend().admit(plan(), Resources(max_usd=cap))


@pytest.mark.parametrize(
    ("reference", "named"),
    [
        ("vastai/base-image:cuda-13.3.1-auto", 13.3),
        ("vastai/base-image:cuda-12.9.2-auto", 12.9),
        ("nvidia/cuda:12.4.1-devel-ubuntu22.04", 12.4),
        ("nvidia/cuda:13.0.2-runtime-ubuntu24.04", 13.0),
        ("pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime", 13.2),
        ("nvcr.io/nvidia/pytorch:25.06-py3", None),
        ("debian:bookworm", None),
    ],
)
def test_image_cuda_reads_the_toolchain_a_reference_names(
    reference: str, named: float | None
) -> None:
    """Every image spelling this house rents, plus two that name no CUDA at all."""
    assert image_cuda(reference) == named


@pytest.mark.parametrize(
    ("reference", "refused"),
    [
        pytest.param("nvidia/cuda:12.4.1-devel-ubuntu22.04", True, id="below-the-floor"),
        pytest.param("vastai/base-image:cuda-13.3.1-auto", False, id="at-the-floor"),
        pytest.param("nvcr.io/nvidia/pytorch:25.06-py3", False, id="naming-no-cuda-at-all"),
    ],
)
def test_admit_refuses_an_image_only_when_it_can_prove_it_below_the_cuda_floor(
    reference: str, refused: bool
) -> None:
    """A retired toolchain is refused naming both versions; an unreadable tag is never refused.

    A provider takes the rent, fails to start a container its driver cannot load, and bills for
    the boot, while an NGC calendar tag or a CPU job carries no evidence either way.
    """
    floor = f"names CUDA 12.4, below this house's CUDA {ProviderBackend.CUDA_FLOOR} floor"
    with pytest.raises(MissionError, match=floor) if refused else nullcontext():
        BareBackend().admit(plan(container=Container(image=reference)), Resources(max_usd=5.0))
