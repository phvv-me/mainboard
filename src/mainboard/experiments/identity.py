# The two content-hash identities a study is built from: `run_id` per trial config, `study_id`
# above the whole trial set. Both are pure, so the same input resolves to the same id on any host.

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


def study_label(study: str, *, trial: str = "") -> str:
    """The dispatch label a study's trials carry, the join key a report reads back.

    Dispatch keeps it as free text and never parses it; this function and `labelled_study` are
    the only places the `study:<id>[/<trial>]` shape is spelled out.
    """
    return f"study:{study}/{trial}" if trial else f"study:{study}"


def labelled_study(label: str) -> str:
    """The study id inside a dispatch `label`, empty when the label names no study."""
    if not label.startswith("study:"):
        return ""
    return label.removeprefix("study:").split("/", maxsplit=1)[0]


def labelled_trial(label: str) -> str:
    """The trial name inside a dispatch `label`, empty when the label names no trial."""
    if not labelled_study(label):
        return ""
    return label.removeprefix("study:").partition("/")[2]


def study_id(
    *, experiment: str, config_space: Mapping[str, object], source_digest: str
) -> tuple[str, str]:
    """`(id, slug)`: a 12-hex SHA-256 over the experiment, config space and source digest, and
    `f"{experiment}-{id[:6]}"` for filenames and logs.

    Re-running the same study, even from a fresh process on another host, joins the same ledger
    instead of minting a duplicate.

    config_space: the searched fields and their domains.
    source_digest: the content digest of the captured source bundle.
    """
    payload = f"{experiment}:{_sha256(config_space)}:{source_digest}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return digest, f"{experiment}-{digest[:6]}"
