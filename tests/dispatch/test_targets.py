from typing import TYPE_CHECKING

import pytest

from mainboard.dispatch import Facts, resolve, smallest_fit, ssh_hosts
from mainboard.dispatch.targets import CAPABILITIES, find_root, probe_capabilities
from mainboard.manifest import HostProfile

if TYPE_CHECKING:
    from pathlib import Path

from .conftest import machine_with

# --- ssh_hosts ---


def test_ssh_hosts_lists_concrete_aliases_dropping_patterns(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.write_text("Host gold\n  HostName 1.2.3.4\nHost dl*\nHost *\nHost crimson gold\n")
    assert ssh_hosts(config) == ["gold", "crimson"]


def test_ssh_hosts_returns_empty_when_the_config_is_missing(tmp_path: Path) -> None:
    assert ssh_hosts(tmp_path / "nope") == []


# --- find_root ---


def test_find_root_runs_the_root_finder_script() -> None:
    remote = machine_with("/work/x/projects\n")
    assert find_root(remote) == "/work/x/projects"
    assert remote.calls[-1][0] == "bash"


# --- probe_capabilities ---


def test_probe_capabilities_parses_key_value_lines_with_a_gpu() -> None:
    output = """root=/work/x/projects
kind=pbs
gpu=NVIDIA H100, 81920
mem=131072000
account=labgrp
queue=interact-g
pixi=/home/me/.pixi/bin/pixi
uv=/home/me/.local/bin/uv
platform=Linux aarch64
"""
    remote = machine_with(output)
    facts = probe_capabilities(remote, "miyabi-g")
    assert facts.name == "miyabi-g"
    assert facts.root == "/work/x/projects"
    assert facts.kind == "pbs"
    assert facts.gpu_name == "NVIDIA H100"
    assert facts.gpu_mem_mb == 81920
    assert facts.sysmem_gb == round(131072000 / 1024**2)
    assert facts.account == "labgrp"
    assert facts.queue == "interact-g"
    assert facts.pixi == "/home/me/.pixi/bin/pixi"
    assert facts.uv == "/home/me/.local/bin/uv"
    assert facts.platform == "Linux aarch64"


def test_probe_capabilities_without_a_gpu_or_engines_leaves_them_unset() -> None:
    output = (
        "root=~/projects\nkind=ssh\ngpu=\nmem=16777216\naccount=me\nqueue=\n"
        "pixi=\nuv=\nplatform=Darwin arm64\n"
    )
    remote = machine_with(output)
    facts = probe_capabilities(remote, "gold")
    assert facts.gpu_name is None
    assert facts.gpu_mem_mb is None
    assert not facts.pixi
    assert not facts.uv
    assert facts.platform == "Darwin arm64"


def test_capabilities_script_probes_every_reported_field() -> None:
    for field in ("root=", "kind=", "gpu=", "mem=", "account=", "queue=", "pixi=", "uv="):
        assert field in CAPABILITIES
    assert "uname -sm" in CAPABILITIES


# --- Facts ---


def test_facts_vram_prefers_gpu_over_system_memory() -> None:
    facts = Facts(name="h", gpu_mem_mb=1024, sysmem_gb=64)
    assert facts.vram_gb == pytest.approx(1.0)


def test_facts_vram_falls_back_to_system_memory() -> None:
    facts = Facts(name="h", sysmem_gb=64)
    assert facts.vram_gb == pytest.approx(64.0)


def test_facts_vram_is_none_when_unprobed() -> None:
    assert Facts(name="h").vram_gb is None


def test_facts_fits_checks_against_vram() -> None:
    facts = Facts(name="h", sysmem_gb=64)
    assert facts.fits(32)
    assert not facts.fits(128)


def test_facts_fits_is_false_when_unprobed() -> None:
    assert not Facts(name="h").fits(1)


# --- resolve ---


def test_resolve_fills_only_the_unset_profile_fields() -> None:
    profile = HostProfile(kind="auto", root="", account="")
    facts = Facts(name="gold", root="/work/x/projects", kind="pbs", account="labgrp")
    merged = resolve(profile, facts)
    assert merged.kind == "pbs"
    assert merged.root == "/work/x/projects"
    assert merged.account == "labgrp"


def test_resolve_keeps_explicit_manifest_values() -> None:
    profile = HostProfile(kind="ssh", root="/custom/root", account="declared")
    facts = Facts(name="gold", root="/work/x/projects", kind="pbs", account="labgrp")
    merged = resolve(profile, facts)
    assert merged.kind == "ssh"
    assert merged.root == "/custom/root"
    assert merged.account == "declared"


def test_resolve_returns_the_same_profile_when_nothing_needs_filling() -> None:
    profile = HostProfile(kind="ssh", root="/custom/root", account="declared")
    facts = Facts(name="gold")
    assert resolve(profile, facts) is profile


# --- smallest_fit ---


def test_smallest_fit_picks_the_smallest_satisfying_candidate() -> None:
    small = Facts(name="a", sysmem_gb=24)
    big = Facts(name="b", sysmem_gb=80)
    assert smallest_fit([small, big], 20).name == "a"
    assert smallest_fit([small, big], 40).name == "b"


def test_smallest_fit_raises_with_no_candidate_fitting() -> None:
    facts = [Facts(name="a", sysmem_gb=24)]
    with pytest.raises(LookupError, match="no target fits"):
        smallest_fit(facts, 999)
