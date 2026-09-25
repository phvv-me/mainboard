from pathlib import Path

import pytest

from mainboard.engines.compile.pixi_lock import canonical, packages
from mainboard.engines.compile.prefixes import digest_of

# The two spellings one lock had on 2026-09-05, cut to the roster and one package apiece: pixi
# 0.77 labels variants `p1`, `p2` (sorting bare `osx-arm64` first), 0.79 keeps the manifest's
# names. Nothing else moved; the full 1.1 MB pair canonicalises to one text and one digest.
_LINUX = """  virtual-packages:
  - __cuda=13.0
  - __glibc=2.34
  - __unix=0=0
  - __linux=4.18
"""
_OSX = """- name: osx-arm64
  virtual-packages:
  - __unix=0=0
  - __osx=13.0
"""
_ARM = "https://conda.anaconda.org/conda-forge/linux-aarch64/bzip2-1.0.8-h4777abc_10.conda"
_X86 = "https://conda.anaconda.org/conda-forge/linux-64/bzip2-1.0.8-hda65f42_10.conda"
_MAC = "https://conda.anaconda.org/conda-forge/osx-arm64/bzip2-1.0.8-h4e30115_10.conda"

SOLVED_BY_0_77 = f"""version: 7
platforms:
{_OSX}- name: p1
  subdir: linux-64
{_LINUX}- name: p2
  subdir: linux-aarch64
{_LINUX}environments:
  default:
    channels:
    - url: https://conda.anaconda.org/conda-forge/
    packages:
      osx-arm64:
      - conda: {_MAC}
      p1:
      - conda: {_X86}
      p2:
      - conda: {_ARM}
packages:
- conda: {_X86}
  name: bzip2
"""

SOLVED_BY_0_79 = f"""version: 7
platforms:
- name: linux-64-system
  subdir: linux-64
{_LINUX}- name: linux-aarch64-system
  subdir: linux-aarch64
{_LINUX}{_OSX}environments:
  default:
    channels:
    - url: https://conda.anaconda.org/conda-forge/
    packages:
      linux-64-system:
      - conda: {_X86}
      linux-aarch64-system:
      - conda: {_ARM}
      osx-arm64:
      - conda: {_MAC}
packages:
- conda: {_X86}
  name: bzip2
"""


def artifact(where: Path, lock: str) -> Path:
    """A compiled artifact whose manifest is fixed and whose lock is `lock`."""
    where.mkdir(parents=True, exist_ok=True)
    (where / "pixi.toml").write_text('[workspace]\nname = "w"\n', encoding="utf-8")
    (where / "pixi.lock").write_text(lock, encoding="utf-8")
    return where


def test_two_pixis_writing_one_lock_reach_one_environment_address(tmp_path: Path) -> None:
    assert SOLVED_BY_0_77 != SOLVED_BY_0_79
    assert canonical(SOLVED_BY_0_77) == canonical(SOLVED_BY_0_79)
    assert digest_of(artifact(tmp_path / "here", SOLVED_BY_0_77)) == digest_of(
        artifact(tmp_path / "there", SOLVED_BY_0_79)
    )


def test_a_lock_that_changed_what_it_installs_is_still_a_different_environment(
    tmp_path: Path,
) -> None:
    moved = SOLVED_BY_0_79.replace("bzip2-1.0.8", "bzip2-1.0.9")
    dropped = SOLVED_BY_0_79.replace(f"      - conda: {_ARM}\n", "")

    addresses = {
        digest_of(artifact(tmp_path / name, lock))
        for name, lock in (
            ("same", SOLVED_BY_0_79),
            ("moved", moved),
            ("dropped", dropped),
        )
    }

    assert len(addresses) == 3


def test_two_variants_of_one_platform_stay_two_platforms(tmp_path: Path) -> None:
    """Two entries sharing a subdirectory are told apart by the floors they vouch."""
    twin = SOLVED_BY_0_77.replace(
        "- name: p2\n  subdir: linux-aarch64\n", "- name: p2\n  subdir: linux-64\n"
    )
    raised = twin.replace("  - __cuda=13.0\n  - __glibc=2.34\n", "  - __cuda=13.2\n", 1)

    assert canonical(twin).count("- name: linux-64#") == 2
    assert digest_of(artifact(tmp_path / "twin", twin)) != digest_of(
        artifact(tmp_path / "raised", raised)
    )


def test_a_roster_line_naming_no_platform_keeps_its_place_at_the_top() -> None:
    stray = SOLVED_BY_0_79.replace("platforms:\n", "platforms:\n# edited by hand\n", 1)

    assert canonical(stray) == canonical(SOLVED_BY_0_79).replace(
        "platforms:\n", "platforms:\n# edited by hand\n", 1
    )


@pytest.mark.parametrize(
    "lock",
    [
        pytest.param("version: 7\npackages: []\n", id="a-lock-that-names-no-platform"),
        pytest.param("", id="nothing-at-all"),
    ],
)
def test_a_lock_with_no_platform_roster_is_its_own_canonical_form_and_installs_nothing(
    lock: str,
) -> None:
    assert canonical(lock) == lock
    assert packages(lock) == {}


@pytest.mark.parametrize("lock", [SOLVED_BY_0_77, SOLVED_BY_0_79], ids=["pixi 0.77", "pixi 0.79"])
def test_the_packages_a_lock_installs_are_keyed_by_subdirectory_whatever_pixi_labelled_them(
    lock: str,
) -> None:
    assert packages(lock) == {
        "linux-64": [_X86],
        "linux-aarch64": [_ARM],
        "osx-arm64": [_MAC],
    }


def test_every_environment_and_every_kind_of_location_counts_toward_its_platform() -> None:
    """An undeclared platform keeps its label, an empty one is locked but empty, and a stray
    line or a PyPI entry's extras never count."""
    wheel = "https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp314-linux_x86_64.whl"
    lock = SOLVED_BY_0_79.replace(
        "packages:\n- conda:",
        f"""  dev:
    packages:
      - conda: a-stray-line-before-any-label
      linux-64-system:
      - pypi: {wheel}
        extras:
        - cuda: an-extra-that-is-no-package
      win-64:
      - conda: https://conda.anaconda.org/conda-forge/win-64/bzip2.conda
      osx-64:
packages:
- conda:""",
        1,
    )

    found = packages(lock)

    assert found["linux-64"] == [_X86, wheel]
    assert found["win-64"] == ["https://conda.anaconda.org/conda-forge/win-64/bzip2.conda"]
    assert found["osx-64"] == []
