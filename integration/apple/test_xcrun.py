import json
import re

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.apple import Apple
from mainboard.core import ToolOutput

from .builders import devices_reply
from .constants import SIMCTL_LIST
from .fakes import ResultTool

_PHONE = {"name": "Same name", "udid": "device-a", "state": "Shutdown", "isAvailable": True}
_TEXT = st.text(alphabet="ab日", min_size=1, max_size=2)
_DEVICES = st.fixed_dictionaries(
    {"name": _TEXT, "udid": _TEXT, "state": _TEXT, "isAvailable": st.booleans()}
)
_CUT_SHORT = "no complete output; check Xcode and DEVELOPER_DIR"
_VERBS = [
    pytest.param(Apple.results, "xcresulttool get test-results summary", id="results"),
    pytest.param(Apple.diagnostics, "xcresulttool get build-results --path", id="diagnostics"),
    pytest.param(
        lambda apple, _bundle: apple.simulators(), "simctl list devices available", id="simulators"
    ),
]


@pytest.mark.parametrize(
    ("reply", "detail"),
    [
        pytest.param(
            ToolOutput(returncode=127, stderr="Xcode missing"),
            "exit 127, truncated=False): Xcode missing",
            id="the-tool-is-missing",
        ),
        pytest.param(
            ToolOutput(returncode=124, truncated=True),
            f"exit 124, truncated=True): {_CUT_SHORT}",
            id="the-tool-timed-out-without-output",
        ),
        pytest.param(
            ToolOutput(stdout='{"result":"Passed"}', truncated=True),
            f"exit 0, truncated=True): {_CUT_SHORT}",
            id="a-valid-looking-payload-cut-short",
        ),
    ],
)
@pytest.mark.parametrize(("read", "tool"), _VERBS)
def test_a_failed_or_truncated_native_tool_is_never_read_as_evidence(
    bundle, read, tool, reply, detail
) -> None:
    fake = ResultTool(reply=reply)
    with pytest.raises(MissionError, match=re.escape(f"xcrun {tool} failed ({detail}")):
        read(Apple(fake), bundle)
    assert len(fake.calls) == 1


@settings(max_examples=200)
@given(inventory=st.dictionaries(_TEXT, st.lists(_DEVICES, max_size=4), max_size=3))
@example(
    inventory={
        "runtime-new": [_PHONE, {**_PHONE, "udid": "gone", "isAvailable": False}],
        "runtime-old": [{**_PHONE, "udid": "device-b", "state": "Booted"}],
    }
)
@example(inventory={})
def test_simulators_list_every_available_device_under_its_runtime_in_a_stable_order(
    inventory,
) -> None:
    tool = ResultTool(reply=devices_reply(inventory))
    listed = [
        (found["runtime"], found["name"], found["udid"], found["state"])
        for found in Apple(tool).simulators()
    ]
    available = [
        (runtime, device["name"], device["udid"], device["state"])
        for runtime, devices in inventory.items()
        for device in devices
        if device["isAvailable"]
    ]
    assert sorted(listed) == sorted(available)
    assert [row[:3] for row in listed] == sorted(row[:3] for row in listed)
    assert tool.calls == [SIMCTL_LIST]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("not JSON", id="text-that-is-not-json"),
        pytest.param("{}", id="an-empty-object"),
        pytest.param('{"devices": []}', id="devices-as-a-list"),
        pytest.param(
            '{"devices": {"runtime": [{"name": "Device", "udid": "id", "state": "Booted"}]}}',
            id="a-device-without-availability",
        ),
        pytest.param(
            '{"devices": {"runtime": [{"name": "Device", "udid": "id", '
            '"state": "Booted", "isAvailable": "true"}]}}',
            id="availability-spelled-as-text",
        ),
    ],
)
def test_a_malformed_simulator_inventory_is_refused(payload: str) -> None:
    fake = ResultTool(reply=ToolOutput(stdout=payload))
    with pytest.raises(MissionError, match="invalid simctl device inventory"):
        Apple(fake).simulators()
    assert fake.calls == [SIMCTL_LIST]


@pytest.mark.parametrize(
    ("runtimes", "printed"),
    [
        pytest.param({}, [], id="an-empty-inventory-is-valid"),
        pytest.param(
            {"runtime": [_PHONE, {**_PHONE, "udid": "gone", "isAvailable": False}]},
            [{"runtime": "runtime", "name": "Same name", "udid": "device-a", "state": "Shutdown"}],
            id="unavailable-devices-are-not-listed",
        ),
    ],
)
def test_the_simulators_verb_prints_the_available_devices_and_only_reads(
    cli, runtimes, printed
) -> None:
    tool = ResultTool(reply=devices_reply(runtimes))
    code, out = cli(Apple(tool), "simulators", "--json")
    assert (code, json.loads(out)) == (0, printed)
    assert tool.calls == [SIMCTL_LIST]
