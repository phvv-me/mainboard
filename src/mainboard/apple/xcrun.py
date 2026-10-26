"""The selected Xcode's tools, reached through `xcrun` with bounded output."""

from pydantic import ValidationError

from ..core.errors import MissionError
from ..core.execution.tool import Runner, ToolRunner
from .evidence import SimulatorInventory


def bounded(runner: Runner, deadline: float) -> Runner:
    """`runner` stopping every tool at the monotonic `deadline`; a runner of another kind as is."""
    match runner:
        case ToolRunner():
            return runner.model_copy(update={"deadline": deadline})
        case _:
            return runner


def simulators(runner: Runner) -> list[dict[str, str]]:
    """The available devices, never booting, creating or erasing a simulator."""
    payload = xcrun(runner, "simctl", "list", "devices", "available", "--json")
    try:
        inventory = SimulatorInventory.model_validate_json(payload)
    except ValidationError as fault:
        raise MissionError(f"invalid simctl device inventory: {fault}") from fault
    return [
        {"runtime": runtime, "name": device.name, "udid": device.udid, "state": device.state}
        for runtime, devices in sorted(inventory.devices.items())
        for device in sorted(devices, key=lambda item: (item.name, item.udid))
        if device.available
    ]


def xcrun(runner: Runner, *arguments: str) -> str:
    """One tool's complete standard output, refusing a failure or a truncation."""
    found = runner.run("/usr/bin/xcrun", arguments)
    if found.returncode or found.truncated:
        raise MissionError(
            f"xcrun {' '.join(arguments[:4])} failed "
            f"(exit {found.returncode}, truncated={found.truncated}): "
            f"{found.stderr.strip() or 'no complete output; check Xcode and DEVELOPER_DIR'}"
        )
    return found.stdout
