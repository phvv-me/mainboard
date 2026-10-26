"""An offline gpuhunt: one recorded vast.ai offer, each query's set filters appended to a file.

Importing this module installs it; the log is the file `MB_GATE_CATALOG` names.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import gpuhunt

_CALLS = Path(os.environ["MB_GATE_CATALOG"])
_OFFER = SimpleNamespace(
    provider="vastai",
    gpu_name="RTX4090",
    gpu_count=1,
    price=0.5,
    spot=False,
    location="fixture",
    instance_name="offline-4090",
)


def _query(**filters: str | int | list[str] | None) -> list[SimpleNamespace]:
    with _CALLS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({k: v for k, v in filters.items() if v is not None}) + "\n")
    return [_OFFER]


vars(gpuhunt)["query"] = _query
