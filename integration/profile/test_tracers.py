import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from ..builders import run_with_libraries
from .constants import OS_SIGNPOST, REGISTERED, ROCTX, ROCTX_RANGES, SIGNPOST_INTERVALS

_ROCTX = {"roctx.py": ROCTX}
_SIGNPOST = {"os_signpost/__init__.py": OS_SIGNPOST}


@pytest.mark.parametrize(
    ("libraries", "added"),
    [
        pytest.param({}, [], id="no-vendor-library"),
        pytest.param(_ROCTX, ["RoctxTracer"], id="only-roctx"),
        pytest.param(_SIGNPOST, ["SignpostTracer"], id="only-os-signpost"),
        pytest.param({**_ROCTX, **_SIGNPOST}, ["RoctxTracer", "SignpostTracer"], id="both"),
    ],
)
def test_a_vendor_tracer_registers_only_where_its_library_imports(
    tmp_path: Path, libraries: Mapping[str, str], added: list[str]
) -> None:
    assert json.loads(run_with_libraries(tmp_path, libraries, REGISTERED)) == sorted(
        ["NvtxTracer", *added]
    )


def test_the_signpost_tracer_closes_each_interval_with_the_closer_the_library_returned(
    tmp_path: Path,
) -> None:
    assert json.loads(run_with_libraries(tmp_path, _SIGNPOST, SIGNPOST_INTERVALS)) == [
        ["init", "me.phvv.mainboard", "PointsOfInterest"],
        ["begin", "outer"],
        ["end", "outer"],
        ["begin", "detached"],
        ["end", "detached"],
    ]


def test_the_roctx_tracer_stops_each_range_it_started_by_id(
    tmp_path: Path,
) -> None:
    assert json.loads(run_with_libraries(tmp_path, _ROCTX, ROCTX_RANGES)) == [
        ["start", "outer"],
        ["stop", 1],
        ["start", "detached"],
        ["stop", 3],
    ]
