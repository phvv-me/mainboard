from typing import TYPE_CHECKING

import pytest

from mainboard import Manifest, load
from mainboard.manifest import Header, HostProfile, Observe, QueuePolicy, Sync

from ..conftest import MANIFEST

if TYPE_CHECKING:
    from pathlib import Path


def test_profiles_inherit_defaults_field_by_field(workspace: Path) -> None:
    manifest = load(workspace / MANIFEST)
    gold = manifest.profile("gold")
    assert gold.kind == "ssh" and gold.env == "serving"
    assert gold.sync.include == ["packages"]
    assert gold.defaults.walltime == "00:30:00"


def test_sync_layers_union_instead_of_replacing(workspace: Path) -> None:
    miyabi = load(workspace / MANIFEST).profile("miyabi-g")
    assert miyabi.sync.include == ["packages"]
    assert miyabi.sync.exclude == ["data/raw"]
    assert miyabi.sync.protect == ["results/***"]


def test_queue_policies_enforce_walltime_ceilings(workspace: Path) -> None:
    miyabi = load(workspace / MANIFEST).profile("miyabi-g")
    short = miyabi.policy("short-g")
    assert short.admits_walltime("06:00:00")
    assert not short.admits_walltime("08:00:00")
    assert miyabi.policy("unknown-queue").admits_walltime("99:00:00")


def test_undeclared_host_gets_the_defaults_profile(workspace: Path) -> None:
    stranger = load(workspace / MANIFEST).profile("stranger")
    assert stranger.kind == "auto"
    assert stranger.sync.protect == ["results/***"]


def test_host_references_must_resolve() -> None:
    with pytest.raises(ValueError, match="names container 'ghost'"):
        Manifest(
            workspace=Header(name="lab"),
            hosts={"gold": HostProfile(container="ghost")},
        )
    with pytest.raises(ValueError, match="names environment 'ghost'"):
        Manifest(
            workspace=Header(name="lab"),
            hosts={"gold": HostProfile(env="ghost")},
        )


def test_inheriting_merges_modules_vars_and_queues() -> None:
    base = HostProfile(
        modules={"cuda": "13.0"},
        vars={"a": "1"},
        queues={"q": QueuePolicy(max_walltime="01:00:00")},
        sync=Sync(protect=["results/***"]),
    )
    child = HostProfile(modules={"singularity": "4.2.1"}, vars={"b": "2"})
    landed = child.inheriting(base)
    assert landed.modules == {"cuda": "13.0", "singularity": "4.2.1"}
    assert landed.vars == {"a": "1", "b": "2"}
    assert landed.policy("q").max_walltime == "01:00:00"
    assert landed.sync.protect == ["results/***"]


def test_host_profile_defaults_to_a_polling_observe_posture() -> None:
    observe = HostProfile().observe
    assert observe == Observe(level="poll", channel="auto", poll_seconds=30.0)


def test_undeclared_host_from_the_manifest_gets_the_default_observe_posture(
    workspace: Path,
) -> None:
    gold = load(workspace / MANIFEST).profile("gold")
    assert gold.observe == Observe()


def test_a_hosts_own_observe_wins_wholesale_over_the_defaults_profile() -> None:
    base = HostProfile(observe=Observe(level="stream", channel="stream", poll_seconds=5.0))
    child = HostProfile(observe=Observe(level="off"))
    landed = child.inheriting(base)
    assert landed.observe == Observe(level="off")


def test_a_host_with_no_observe_of_its_own_inherits_the_defaults_profiles() -> None:
    base = HostProfile(observe=Observe(level="interactive", channel="stream"))
    landed = HostProfile().inheriting(base)
    assert landed.observe == base.observe
