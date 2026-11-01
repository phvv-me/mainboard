import json
from pathlib import Path

from ...builders import run_with_libraries
from .constants import ABSENT, MASKED_READINGS, NVML, PACKAGES, READINGS, RUNTIME


def test_gpus_are_read_through_the_runtime_and_nvml_when_both_import(
    tmp_path: Path,
) -> None:
    libraries = {**PACKAGES, "cuda/bindings/nvml.py": NVML, "cuda/bindings/runtime.py": RUNTIME}
    assert json.loads(run_with_libraries(tmp_path, libraries, READINGS)) == {
        "labels": ["Fake GPU 100", "Fake GPU 101"],
        "bus_ids": ["0000:00:00.0", "0000:01:00.0"],
        "arch": ["sm_89", "sm_89"],
        "coherent": [True, True],
        "runtime": [[13, 3], [13, 3]],
        "driver": "999.1",
        "temperature": 61,
        "throttles": ["power cap", "hardware slowdown"],
    }


def test_a_visible_devices_mask_picks_the_nvml_handle_when_the_runtime_is_absent(
    tmp_path: Path,
) -> None:
    libraries = {**PACKAGES, "cuda/bindings/nvml.py": NVML}
    assert json.loads(run_with_libraries(tmp_path, libraries, MASKED_READINGS)) == {
        "labels": ["Fake GPU 101"],
        "bus_ids": ["0000:01:00.0"],
        "runtime": [None],
    }


def test_no_gpu_is_found_where_nvml_does_not_import(
    tmp_path: Path,
) -> None:
    assert json.loads(run_with_libraries(tmp_path, PACKAGES, ABSENT)) == [False, 0]
