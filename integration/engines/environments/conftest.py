from pathlib import Path

import pytest

from mainboard.engines.compile import Lockfile
from mainboard.engines.compile.pixi.lockfile import Solved

_LOCK = """version: 7
platforms:
- name: linux-64-system
  subdir: linux-64
- name: linux-aarch64
- name: osx-arm64
environments:
  default:
    packages:
      linux-64-system:
      - conda: https://example.invalid/linux-64/python-3.14.8-build_cp314.conda
      - conda: https://example.invalid/linux-64/python_abi-3.14-8_cp314.conda
      - pypi: https://files.pythonhosted.org/packages/aa/bb/duckdb-2.0.0-cp312-abi3-manylinux_2_27_x86_64.whl
      - pypi: https://files.pythonhosted.org/packages/cc/dd/pytz-2026.1.tar.gz
      - pypi: https://pypi.nvidia.com/cuda-bindings/cuda_bindings-13.3.1-cp314-cp314-manylinux_2_28_x86_64.whl
      - pypi: ./packages/patos
      linux-aarch64:
      - conda: https://example.invalid/linux-aarch64/python-3.15.1-build_cp315.tar.bz2
      osx-arm64:
      - conda: https://example.invalid/osx-arm64/python-3.16.0rc1-build_cp316.conda
  training:
    packages:
      linux-64-system:
      - conda: https://example.invalid/linux-64/python-3.17.0-build_cp317.conda
"""


@pytest.fixture
def locked(workspace: Path) -> Path:
    """A workspace whose committed lock solves `default`, with a Python per platform.

    The lock text also solves `training` with a Python of its own, which no tool should follow.
    """
    lock = Lockfile(workspace)
    solved = Solved(solved_from="fake", solved_by="fake", lock=_LOCK)
    lock.path.write_text(lock.render({"default": solved}), encoding="utf-8")
    return workspace
