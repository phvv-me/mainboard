import json
from functools import cache

from plumbum import CommandNotFound, local

# A decoded JSON value, and one `system_profiler` record (a flat-ish JSON object).
type Json = str | int | float | bool | None | list["Json"] | dict[str, "Json"]
type ProfileRecord = dict[str, Json]
type SystemProfile = dict[str, list[ProfileRecord]]


@cache
def system_profiler(*datatypes: str) -> SystemProfile:
    """Parsed macOS `system_profiler` data for one or more datatypes.

    Runs `system_profiler <datatypes...> -json` once per datatype set and parses
    the result. Returns an empty mapping when the tool is missing or its output
    cannot be parsed, so callers stay platform-tolerant.

    datatypes: profiler datatype keys, e.g. `"SPHardwareDataType"`.
    """
    try:
        parsed: SystemProfile = json.loads(local["system_profiler"]("-json", *datatypes))
        return parsed
    except CommandNotFound, OSError, json.JSONDecodeError:
        return {}
