import pytest

from mainboard.engines.compile.pixi.lock import canonical

_PIXI_0_77 = """\
platforms:
- name: p3
  subdir: osx-arm64
- name: p2
  subdir: linux-64
  virtual-packages:
  - __cuda=13.0
- name: p1
  subdir: linux-64
  virtual-packages:
  - __cuda=12.4
environments:
  default:
    packages:
      p3:
      - conda: https://example.invalid/a.conda
      p2:
      - conda: https://example.invalid/b.conda
      p1:
      - conda: https://example.invalid/c.conda
"""

_PIXI_0_79 = (
    _PIXI_0_77.replace("p3", "osx-arm64-system")
    .replace("p2", "linux-64-system")
    .replace("p1", "linux-64-cuda12")
)

_CANONICAL = """\
platforms:
- name: linux-64#0
  subdir: linux-64
  virtual-packages:
  - __cuda=12.4
- name: linux-64#1
  subdir: linux-64
  virtual-packages:
  - __cuda=13.0
- name: osx-arm64
  subdir: osx-arm64
environments:
  default:
    packages:
      linux-64#0:
      - conda: https://example.invalid/c.conda
      linux-64#1:
      - conda: https://example.invalid/b.conda
      osx-arm64:
      - conda: https://example.invalid/a.conda
"""

_NO_PACKAGES = (
    "platforms:\n- name: p1\n  subdir: linux-64\nenvironments:\n  default:\n    packages:\n"
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(_PIXI_0_77, _CANONICAL, id="pixi-0-77-labels"),
        pytest.param(_PIXI_0_79, _CANONICAL, id="pixi-0-79-labels"),
        pytest.param(_NO_PACKAGES, _NO_PACKAGES.replace("p1", "linux-64"), id="an-empty-mapping"),
        pytest.param("version: 7\n", "version: 7\n", id="a-lock-with-no-platforms"),
        pytest.param("platforms: [p1\n", "platforms: [p1\n", id="text-that-is-not-yaml"),
    ],
)
def test_the_canonical_lock_names_platforms_by_subdirectory_whatever_pixi_labelled_them(
    text: str, expected: str
) -> None:
    assert canonical(text) == expected
