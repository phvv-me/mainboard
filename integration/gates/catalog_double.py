import json
from pathlib import Path
from types import SimpleNamespace

import gpuhunt


def install(calls: Path) -> None:
    def query(**filters: str | int | list[str]) -> list[SimpleNamespace]:
        with calls.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(filters) + "\n")
        return [SimpleNamespace(
            provider="vastai", gpu_name="RTX4090", gpu_count=1, price=0.5,
            spot=False, location="fixture", instance_name="offline-4090",
        )]

    gpuhunt.query = query
