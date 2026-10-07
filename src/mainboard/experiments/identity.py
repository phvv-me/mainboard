# The content-hash identity of one run's config, pure, so the same config resolves to the same id
# on any host.

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def _sha256(mapping: Mapping[str, object]) -> str:
    """The hex sha256 of `mapping`'s canonical JSON, whose sorted keys ignore declaration order."""
    return hashlib.sha256(
        json.dumps(dict(mapping), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_id(config: Mapping[str, object]) -> str:
    """The dedup key for one trial's JSON-native config: its canonical sha256, first 16 hex chars.

    Byte-for-byte `research/common/experiments/experiment.py`'s `Experiment.run_id`, so a run
    dispatched through mainboard resolves to the id an in-process research run gives.
    """
    return _sha256(config)[:16]
