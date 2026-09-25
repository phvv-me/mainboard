# How a run's trial receipts (one JSON line each, under the `trial_receipt` key) get off the
# machine that produced them.
#
# On an ssh host the job script's captured log is a file read back whole. A rented instance hands
# back a log instead, and vast truncates every log line at 500 characters (measured live
# 2026-08-25), which cuts a receipt in half: the command succeeded, the rental was paid in full,
# and the trial is silently lost, a worse outcome than a loud failure.
#
# So a run writes its receipts to a file and whatever started it frames that file back through the
# output: the job runner in Python (`framed`) where the tool is installed, a shell pipeline
# (`staging`, `framing`) in a prebuilt container that carries no tool. The frame is base64, so a
# receipt's quoting survives a shell and a log viewer, in fixed-width marked chunk lines narrower
# than the cut, between a begin and an end line. `receipts_in` reads both the framed block and a
# plainly printed line, so one harvest serves a queued job and a rented instance alike, and
# `printed` takes the frame back out for a reader who wants the job's own words.

import base64
import re
import string

from ..jobs.beacon import unbeaconed

# A rented machine keeps no workspace, hence `/tmp`. The shell's pid is in the name because a
# cluster node runs several jobs out of one `/tmp`, and a shared file would hand each other's
# trials to whichever settled first.
RECEIPTS_VAR = "MAINBOARD_RECEIPTS"
RECEIPTS_FILE = "/tmp/mainboard-receipts.$$.ndjson"

# Spelled here rather than imported so harvesting never drags the lab machinery in (as `verdicts`).
_RECEIPT = "trial_receipt"

# Chunks stay narrower than the tightest provider line limit measured, with room for the marker.
_BEGIN = "mainboard-receipts-begin"
_END = "mainboard-receipts-end"
_CHUNK_MARKER = "mainboard-receipt:"
_CHUNK_WIDTH = 240

# Checking a payload's alphabet and quantum skips a torn upload without asking an exception, so a
# bad frame costs its own block and never the receipts printed beside it.
_BASE64_ALPHABET = frozenset(string.ascii_letters + string.digits + "+/=")


def staging() -> str:
    """The shell that points a run at its receipts file and starts it empty.

    Emptying matters on a provider that restarts an exited container, which would otherwise frame
    the first run's receipts again alongside the second's.
    """
    return f"export {RECEIPTS_VAR}={RECEIPTS_FILE}; : > ${RECEIPTS_VAR}"


def framing() -> str:
    """The shell that emits the receipts file back through the run's captured output.

    Uses only tools any Linux image carries, since a rented machine was never provisioned. A run
    that wrote no receipts emits no block, leaving an ordinary log untouched.

    The bare `echo` in the braces is load-bearing: `tr` leaves no trailing newline, so `fold` would
    end its last chunk unterminated, glue the end marker onto it and silently lose the block.
    The `|| true` too: `set -euo pipefail` still applies inside an `if` body, so an image missing
    one tool would kill the script before it reports the command's exit code; evidence must never
    change the outcome it is evidence of. It guards the pipeline rather than the statement because
    a caller redirecting this into a log redirects the `if`, and an outer `|| true` would capture
    that redirect instead.
    """
    file = f'"${RECEIPTS_VAR}"'
    return (
        f"if [ -s {file} ]; then echo {_BEGIN}; "
        f'{{ base64 < {file} | tr -d "\\n"; echo; }} | fold -w {_CHUNK_WIDTH} '
        f"| sed 's/^/{_CHUNK_MARKER}/' || true; echo {_END}; fi"
    )


def framed(receipts: bytes) -> str:
    """`receipts` (never empty) framed exactly as `framing` frames a file."""
    payload = base64.b64encode(receipts).decode("ascii")
    chunks = [payload[at : at + _CHUNK_WIDTH] for at in range(0, len(payload), _CHUNK_WIDTH)]
    return "\n".join([_BEGIN, *(f"{_CHUNK_MARKER}{chunk}" for chunk in chunks), _END]) + "\n"


def unframed(log: str) -> str:
    """The receipts text `log`'s last whole frame carries, empty when it holds none.

    The last, since a restarted container appends one block per run and the newest is the one the
    verdict is about; a block torn mid-upload is skipped for the next one down.
    """
    lines = [line.strip() for line in log.splitlines()]
    for start in reversed([at for at, line in enumerate(lines) if line == _BEGIN]):
        ends = [at for at, line in enumerate(lines[start:], start) if line == _END]
        if not ends:
            continue
        payload = "".join(
            line.removeprefix(_CHUNK_MARKER)
            for line in lines[start + 1 : ends[0]]
            if line.startswith(_CHUNK_MARKER)
        )
        if payload and len(payload) % 4 == 0 and set(payload) <= _BASE64_ALPHABET:
            return base64.b64decode(payload).decode(errors="replace")
    return ""


def printed(log: str) -> str:
    """`log` as the job printed it, without any frame line (torn blocks included) or beacon.

    The frame is this wrapper's channel, read through `verdict` and the receipts files; progress
    beacons are read by `wait` and `jobs`.
    """
    kept = [
        line
        for line in log.splitlines(keepends=True)
        if line.strip() not in (_BEGIN, _END) and not line.lstrip().startswith(_CHUNK_MARKER)
    ]
    return unbeaconed("".join(kept))


# What the trials plugin prints for a cell a previous run already took (the coverage heading and
# the `-rs` skip reason), and the summary such a session ends with. A job made only of those cells
# has nothing to deliver and is still settled.
_COVERED = re.compile(
    r"^\s*complete \S+ on .*\d+/\d+ from \S+|complete, run \S+ took it", re.MULTILINE
)
_ONLY_SKIPS = re.compile(r"^\s*\d+ skipped in [\d.]+s\s*$", re.MULTILINE)
_ACQUIRED = re.compile(r"\b\d+ (passed|known|failed|error)", re.MULTILINE)


def covered_in(log: str) -> bool:
    """Whether `log` is a trials session whose every cell was already complete and skipped."""
    skipped = _COVERED.search(log) is not None and _ONLY_SKIPS.search(log) is not None
    return skipped and _ACQUIRED.search(log) is None


def receipts_in(log: str) -> tuple[str, ...]:
    """Every trial receipt `log` carries, framed or printed, deduplicated in first-seen order.

    A harness that both writes and prints a receipt still ran one trial.
    """
    lines = [*unframed(log).splitlines(), *log.splitlines()]
    found = [line.strip() for line in lines if _RECEIPT in line and line.strip()]
    return tuple(dict.fromkeys(found))
