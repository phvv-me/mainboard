# The two content-hash identities a study is built from: `run_id` per trial config, `study_id`
# above the whole trial set. Neither writes anything, both are pure functions of their inputs, so
# the same config or config space always resolves to the same id, on this host or any other.

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def run_id(config: Mapping[str, object]) -> str:
    """The dedup key for one trial's config: sha256 of its canonical JSON, first 16 hex chars.

    Byte-for-byte compatible with `research/common/experiments/experiment.py`'s
    `Experiment.run_id` (same `json.dumps(config, sort_keys=True, separators=(",", ":"))` input,
    same `hashlib.sha256(...).hexdigest()[:16]`), so a run dispatched through mainboard resolves
    to the same id an in-process research run would give the same config.

    config: the trial's JSON-native field mapping.
    """
    canonical = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def study_label(study: str, *, trial: str = "") -> str:
    """The dispatch label a study's trials carry, the join key a report reads back.

    Dispatch keeps it as free text and never parses it, so this function and `labelled_study`
    are the only two places the `study:` shape is spelled out.

    study: the study id owning the trial.
    trial: an optional suffix distinguishing trials within one study by name.
    """
    return f"study:{study}/{trial}" if trial else f"study:{study}"


def labelled_study(label: str) -> str:
    """The study id inside a dispatch `label`, empty when the label names no study.

    label: a dispatched run's free-text name, `study:<id>` or `study:<id>/<trial>` for a trial.
    """
    if not label.startswith("study:"):
        return ""
    return label.removeprefix("study:").split("/", maxsplit=1)[0]


def study_id(
    *, experiment: str, config_space: Mapping[str, object], git_sha: str
) -> tuple[str, str]:
    """The identity above runs: sha256 over (experiment, sorted config-space digest, git sha).

    Two calls with the same experiment, config space, and git sha always resolve to the same
    id, so re-running the same study (even from a fresh process, even on a different host)
    joins the same ledger instead of minting a duplicate one. Returns `(id, slug)`, the 12-hex
    id plus a human-readable slug (`f"{experiment}-{id[:6]}"`) fit for filenames and logs.

    experiment: the registered experiment name the study runs.
    config_space: the study's config space (its searched fields and domains), hashed with
        sorted keys so field declaration order never changes the id.
    git_sha: the short HEAD sha the study was created at.
    """
    space_digest = hashlib.sha256(
        json.dumps(dict(config_space), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = f"{experiment}:{space_digest}:{git_sha}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return digest, f"{experiment}-{digest[:6]}"
