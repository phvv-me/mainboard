"""An offline gpuhunt: one recorded vast.ai offer, and each query's filters appended to a file.

Importing this module installs it; the log is the file `MB_GATE_CATALOG` names.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import gpuhunt

_CALLS = Path(os.environ["MB_GATE_CATALOG"])
_OFFER = SimpleNamespace(
    provider="vastai", gpu_name="RTX4090", gpu_count=1, price=0.5,
    spot=False, location="fixture", instance_name="offline-4090",
)


def _query(**filters: str | int | list[str]) -> list[SimpleNamespace]:
    with _CALLS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(filters) + "\n")
    return [_OFFER]


gpuhunt.query = _query
