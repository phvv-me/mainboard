from collections.abc import Sequence
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.dispatch import Facts, resolve, smallest_fit, ssh_hosts
from mainboard.dispatch import targets as targets_mod
from mainboard.dispatch.targets import find_root, probe_capabilities
from mainboard.manifest import HostProfile

from ..strategies import WORDS
from .support import RecordingTransport, machine_with

_GPU_PROBE = """root=/work/x/projects
kind=pbs
gpu=NVIDIA H100, 81920
mem=131072000
account=labgrp
queue=interact-g
pixi=/home/me/.pixi/bin/pixi
uv=/home/me/.local/bin/uv
platform=Linux aarch64
"""


@given(aliases=st.lists(WORDS, max_size=4), patterns=st.lists(WORDS, max_size=2))
@example(aliases=["gold", "crimson", "gold"], patterns=["dl"])
@example(aliases=[], patterns=[])
def test_ssh_hosts_keeps_every_concrete_alias_in_file_order_and_drops_the_patterns(
    tmp_path: Path, aliases: list[str], patterns: Sequence[str]
) -> None:
    lines = ["# a comment", "", "  HostName 1.2.3.4", "Host *"]
    lines += [f"Host {pattern}*" for pattern in patterns]
    lines += [f"Host\t{alias}" for alias in aliases]
    config = tmp_path / "config"
    config.write_text("\n".join(lines) + "\n")
    assert ssh_hosts(config) == list(dict.fromkeys(aliases))
    assert ssh_hosts(tmp_path / "nope") == []


def test_a_multi_alias_host_line_yields_each_of_its_destinations(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.write_text("Host gold\n  HostName 1.2.3.4\nHost dl*\nHost *\nHost crimson gold\n")
    assert ssh_hosts(config) == ["gold", "crimson"]


def test_the_capabilities_probe_parses_the_key_value_lines_its_own_script_prints() -> None:
    for field in ("root=", "kind=", "gpu=", "mem=", "account=", "queue=", "pixi=", "uv="):
        assert field in targets_mod._CAPABILITIES
        assert field in targets_mod._WINDOWS_CAPABILITIES
    assert "uname -sm" in targets_mod._CAPABILITIES
    transport = RecordingTransport(rules=[("bash -lc", 0, _GPU_PROBE)])
    facts = probe_capabilities("miyabi-g", ssh=transport)
    assert transport.calls[-1][:3] == ["ssh", "-o", "BatchMode=yes"]
    assert transport.calls[-1][3:5] == ["miyabi-g", "bash"]
    assert facts == Facts(
        name="miyabi-g",
        root="/work/x/projects",
        kind="pbs",
        account="labgrp",
        queue="interact-g",
        gpu_name="NVIDIA H100",
        gpu_mem_mb=81920,
        sysmem_gb=round(131072000 / 1024**2),
        platform="Linux aarch64",
        pixi="/home/me/.pixi/bin/pixi",
        uv="/home/me/.local/bin/uv",
    )
    assert facts.pixi_platform == "linux-aarch64"


def test_a_host_with_no_bash_is_asked_the_same_questions_in_powershell() -> None:
    windows = "root=C:/Users/me/projects\nkind=ssh\ngpu=NVIDIA GeForce RTX 5080, 16303\n"
    windows += "mem=67108864\naccount=\nqueue=\npixi=\nuv=C:\\Users\\me\\.local\\bin\\uv.exe\n"
    windows += "platform=Windows AMD64\n"
    transport = RecordingTransport(
        rules=[("bash -lc", 1, "'bash' is not recognized"), ("Get-CimInstance", 0, windows)]
    )
    facts = probe_capabilities("homelab", ssh=transport)
    assert [argv[4] for argv in transport.calls] == ["bash", "powershell"]
    assert (facts.root, facts.gpu_name, facts.gpu_mem_mb) == (
        "C:/Users/me/projects",
        "NVIDIA GeForce RTX 5080",
        16303,
    )
    assert (facts.platform, facts.pixi_platform) == ("Windows AMD64", "win-64")
    mute = RecordingTransport(rules=[("bash -lc", 1, "no bash"), ("Get-CimInstance", 1, "")])
    with pytest.raises(MissionError, match="neither the bash nor the PowerShell probe: no bash"):
        probe_capabilities("silent", ssh=mute)


def test_a_machine_with_no_gpu_engines_or_readable_memory_leaves_those_facts_unset() -> None:
    bare = "root=~/projects\nkind=ssh\ngpu=\nmem=nonsense\naccount=me\nqueue=\npixi=\nuv=\n"
    facts = Facts.parsed("gold", f"{bare}platform=Darwin arm64\n")
    assert (facts.gpu_name, facts.gpu_mem_mb, facts.sysmem_gb) == (None, None, None)
    assert (facts.pixi, facts.uv, facts.platform) == ("", "", "Darwin arm64")
    assert facts.pixi_platform == "osx-arm64"
    assert find_root(machine_with("/work/x/projects\n")) == "/work/x/projects"


@pytest.mark.parametrize(
    ("facts", "usable", "fits_32"),
    [
        (Facts(name="h", gpu_mem_mb=1024, sysmem_gb=64), 1.0, False),
        (Facts(name="h", sysmem_gb=64), 64.0, True),
        (Facts(name="h"), None, False),
    ],
)
def test_usable_memory_is_the_gpu_when_there_is_one_and_the_system_otherwise(
    facts: Facts, usable: float | None, fits_32: bool
) -> None:
    assert facts.vram_gb == (pytest.approx(usable) if usable is not None else None)
    assert facts.fits(32) is fits_32


def test_resolve_fills_only_the_gaps_the_manifest_left_open() -> None:
    facts = Facts(
        name="gold", root="/work/x/projects", kind="pbs", account="labgrp", platform="Linux x86_64"
    )
    filled = resolve(HostProfile(kind="auto", root="", account=""), facts)
    assert (filled.kind, filled.root, filled.account) == ("pbs", "/work/x/projects", "labgrp")
    assert filled.platform == "linux-64"
    declared = HostProfile(kind="ssh", root="/custom/root", account="declared", platform="win-64")
    kept = resolve(declared, facts)
    assert (kept.kind, kept.root, kept.account) == ("ssh", "/custom/root", "declared")
    assert kept.platform == "win-64"
    assert resolve(declared, Facts(name="gold")) is declared


def test_smallest_fit_keeps_the_big_iron_free_or_names_what_it_had_to_choose_from() -> None:
    candidates = [Facts(name="a", sysmem_gb=24), Facts(name="b", sysmem_gb=80)]
    assert smallest_fit(candidates, 20).name == "a"
    assert smallest_fit(candidates, 40).name == "b"
    with pytest.raises(LookupError, match="no target fits 999 GB; have: a=24.0, b=80.0"):
        smallest_fit(candidates, 999)
