import base64
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mainboard.dispatch.evidence import RECEIPTS_VAR, framing, receipts_in, staging, unframed

# One trial receipt long enough that vast's 500-character line limit would cut it in half, which
# is the whole reason this channel exists.
_RECEIPT = json.dumps({"trial_receipt": {"run_id": "r1", "outcome": "passed", "pad": "x" * 900}})


@pytest.fixture(scope="module")
def posix_bash() -> str:
    """A real POSIX Bash, never Windows' WSL launcher shim.

    GitHub's Windows image puts ``System32/bash.exe`` on PATH even when no WSL distribution
    exists. Git for Windows ships the Bash and coreutils that can actually exercise this remote
    POSIX protocol, so use that installation explicitly instead of trusting the ambiguous name.
    """
    if sys.platform != "win32":
        if bash := shutil.which("bash"):
            return bash
        pytest.skip("remote Bash protocol test needs Bash")

    roots = [Path(git).resolve().parent.parent] if (git := shutil.which("git")) else []
    roots.extend(
        Path(value) / "Git"
        for name in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)")
        if (value := os.environ.get(name))
    )
    if bash := next(
        (root / "bin" / "bash.exe" for root in roots if (root / "bin" / "bash.exe").is_file()),
        None,
    ):
        return str(bash)
    pytest.skip("remote Bash protocol test needs Git for Windows Bash")


def block(receipts: str) -> str:
    """`receipts` framed the way the shell in `framing` frames them, chunk width included.

    Spelled out here rather than imported, so the frame's wire format is pinned by a second
    reading of it and a change to either half has to be argued against the other.
    """
    encoded = base64.b64encode(receipts.encode()).decode()
    chunks = [encoded[at : at + 240] for at in range(0, len(encoded), 240)]
    return "\n".join(
        [
            "mainboard-receipts-begin",
            *(f"mainboard-receipt:{chunk}" for chunk in chunks),
            "mainboard-receipts-end",
        ]
    )


def test_the_staged_variable_and_the_framing_shell_name_the_same_file():
    """A run writes where the export points and the wrapper reads back from the same place."""
    assert staging().startswith(f"export {RECEIPTS_VAR}=/tmp/mainboard-receipts.$$.ndjson")
    assert f": > ${RECEIPTS_VAR}" in staging()
    assert f'"${RECEIPTS_VAR}"' in framing()
    # Narrower than the tightest provider line limit anyone has measured, with room for the
    # marker, which is the one property that makes the frame survive at all.
    assert "fold -w 240" in framing()


def test_a_receipt_too_long_for_one_log_line_survives_the_frame_whole():
    """The defect this channel exists for: a printed receipt arrives cut, a framed one arrives."""
    framed = block(_RECEIPT)
    assert max(len(line) for line in framed.splitlines()) < 500
    assert unframed(framed) == _RECEIPT
    assert receipts_in(framed) == (_RECEIPT,)


@pytest.mark.parametrize(
    ("log", "expected"),
    [
        pytest.param("no frame at all", "", id="a-log-that-carries-none"),
        pytest.param("mainboard-receipts-begin\nnothing closed it", "", id="a-block-cut-off"),
        pytest.param(
            "mainboard-receipts-begin\nmainboard-receipt:!!not base64!!\nmainboard-receipts-end",
            "",
            id="a-payload-that-will-not-decode",
        ),
        pytest.param("mainboard-receipts-begin\nmainboard-receipts-end", "", id="an-empty-block"),
    ],
)
def test_a_frame_that_never_arrived_whole_costs_its_own_block_and_nothing_else(
    log: str, expected: str
):
    """A torn upload is skipped rather than fatal, the same tolerance a torn log already gets."""
    assert unframed(log) == expected


def test_the_last_whole_block_wins_because_a_restarted_container_appends_another():
    """Vast holds an instance at its intended status and re-runs the exited command."""
    first = json.dumps({"trial_receipt": {"run_id": "first"}})
    second = json.dumps({"trial_receipt": {"run_id": "second"}})
    log = f"{block(first)}\nsome output\n{block(second)}"
    assert unframed(log) == second
    # A torn newest block falls through to the newest whole one rather than to nothing.
    torn = (
        f"{block(first)}\nmainboard-receipts-begin\nmainboard-receipt:!!\nmainboard-receipts-end"
    )
    assert unframed(torn) == first


def test_one_harvest_reads_a_framed_receipt_and_a_plainly_printed_one_alike():
    """A queued job prints its receipts and a rental frames them, and a caller never asks which."""
    printed = json.dumps({"trial_receipt": {"run_id": "printed"}})
    framed = json.dumps({"trial_receipt": {"run_id": "framed"}})
    log = f"starting\n{printed}\n{block(framed)}\ndone"
    assert receipts_in(log) == (framed, printed)


def test_a_receipt_that_was_both_written_and_printed_is_still_one_trial():
    """Duplicates collapse, since a harness doing both has still only run the one trial."""
    receipt = json.dumps({"trial_receipt": {"run_id": "once"}})
    assert receipts_in(f"{receipt}\n{block(receipt)}") == (receipt,)


def test_a_log_with_no_receipts_in_it_harvests_nothing():
    assert receipts_in("epoch 1 loss 0.4\nepoch 2 loss 0.3\n") == ()


def framed_by_the_shell(receipts: str, bash: str) -> str:
    """What `staging` and `framing` really emit, run through a real shell rather than described.

    The one test that can catch a fault in the pipeline itself. Everything above reads a frame
    this file built in Python, which proves the parser and proves nothing about the six coreutils
    that produce the frame on a rented machine nobody can attach a debugger to.
    """
    script = f"""{staging()}
printf '%s\\n' {shlex.quote(receipts)} >> "${RECEIPTS_VAR}"
echo "epoch 1 loss 0.4"
{framing()}
"""
    done = subprocess.run([bash, "-c", script], capture_output=True, text=True, check=True)
    return done.stdout


@pytest.mark.parametrize(
    "pad",
    [0, 1, 900, 240 * 3],
    ids=[
        "one-short-receipt",
        "a-length-off-by-one",
        "longer-than-the-500-char-cut",
        "an-exact-multiple-of-the-chunk-width",
    ],
)
def test_the_shell_that_really_runs_on_the_instance_frames_a_receipt_back_whole(
    pad: int, posix_bash: str
):
    """The pipeline is six coreutils deep and runs where nothing can be inspected.

    `tr` leaves the stream unterminated, so `fold` used to end its last chunk without a newline
    and the closing marker landed glued to it, which made the block unreadable and lost every
    receipt in it. A payload that lands on an exact chunk boundary is the case where a frame can
    look right and still be wrong, so it is pinned by name.
    """
    receipt = json.dumps(
        {"trial_receipt": {"run_id": "r1", "outcome": "passed", "pad": "x" * pad}}
    )
    emitted = framed_by_the_shell(receipt, posix_bash)
    assert max(len(line) for line in emitted.splitlines()) < 500
    assert unframed(emitted) == f"{receipt}\n"
    assert receipts_in(emitted) == (receipt,)


@pytest.mark.parametrize("code", [0, 7], ids=["a-command-that-succeeded", "one-that-failed"])
def test_an_image_missing_the_tools_costs_its_receipts_and_never_the_jobs_exit_code(
    code: int, posix_bash: str
):
    """A job script runs under `set -e`, where a failing command in an `if` body is not exempt.

    So a minimal container without `fold` would have taken the whole script down inside the
    framing, before the line that reports what the command itself did. Evidence is worth a great
    deal and never worth changing the outcome it is evidence of.
    """
    script = f"""set -euo pipefail
{staging()}
printf '%s\\n' '{{"trial_receipt": {{"run_id": "r1"}}}}' >> "${RECEIPTS_VAR}"
status=0
bash -c 'exit {code}' || status=$?
PATH=/nonexistent
{framing()}
exit $status
"""
    # `base64`, `tr`, `fold` and `sed` are the four externals here, so emptying PATH after the
    # command has run is exactly an image that never carried them.
    bare = subprocess.run([posix_bash, "-c", script], capture_output=True, text=True, check=False)
    assert bare.returncode == code
    assert receipts_in(bare.stdout) == ()


def test_a_run_that_wrote_no_receipts_leaves_an_ordinary_log_exactly_as_it_was(
    posix_bash: str,
):
    """The framing must be invisible to every command that never writes a trial."""
    script = f'{staging()}\necho "epoch 1 loss 0.4"\n{framing()}\n'
    done = subprocess.run([posix_bash, "-c", script], capture_output=True, text=True, check=True)
    assert done.stdout == "epoch 1 loss 0.4\n"
    assert receipts_in(done.stdout) == ()
