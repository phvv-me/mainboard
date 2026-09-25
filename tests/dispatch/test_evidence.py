import base64
import json
import shlex
import subprocess

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.dispatch.evidence import (
    RECEIPTS_VAR,
    framing,
    printed,
    receipts_in,
    staging,
    unframed,
)

from ..strategies import TEXT

# Long enough that vast's 500-character line limit would cut it in half.
_RECEIPT = json.dumps({"trial_receipt": {"run_id": "r1", "outcome": "passed", "pad": "x" * 900}})


def receipt(run_id: str) -> str:
    return json.dumps({"trial_receipt": {"run_id": run_id}})


def block(receipts: str) -> str:
    """`receipts` framed as `framing` frames them, spelled out so a second reading pins the wire
    format and a change to either half has to be argued against the other."""
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
    assert staging().startswith(f"export {RECEIPTS_VAR}=/tmp/mainboard-receipts.$$.ndjson")
    assert f": > ${RECEIPTS_VAR}" in staging()
    assert f'"${RECEIPTS_VAR}"' in framing()
    assert "fold -w 240" in framing()


def test_a_receipt_too_long_for_one_log_line_survives_the_frame_whole():
    framed = block(_RECEIPT)
    assert max(len(line) for line in framed.splitlines()) < 500
    assert unframed(framed) == _RECEIPT
    assert receipts_in(framed) == (_RECEIPT,)


@pytest.mark.parametrize(
    "log",
    [
        pytest.param("no frame at all", id="a-log-that-carries-none"),
        pytest.param("mainboard-receipts-begin\nnothing closed it", id="a-block-cut-off"),
        pytest.param(
            "mainboard-receipts-begin\nmainboard-receipt:!!not base64!!\nmainboard-receipts-end",
            id="a-payload-that-will-not-decode",
        ),
        pytest.param("mainboard-receipts-begin\nmainboard-receipts-end", id="an-empty-block"),
    ],
)
def test_a_frame_that_never_arrived_whole_costs_its_own_block_and_nothing_else(log: str):
    assert unframed(log) == ""


def test_the_last_whole_block_wins_because_a_restarted_container_appends_another():
    first, second = receipt("first"), receipt("second")
    assert unframed(f"{block(first)}\nsome output\n{block(second)}") == second
    # A torn newest block falls through to the newest whole one rather than to nothing.
    torn = "mainboard-receipts-begin\nmainboard-receipt:!!\nmainboard-receipts-end"
    assert unframed(f"{block(first)}\n{torn}") == first


@pytest.mark.parametrize(
    ("log", "expected"),
    [
        pytest.param(
            f"starting\n{receipt('printed')}\n{block(receipt('framed'))}\ndone",
            (receipt("framed"), receipt("printed")),
            id="framed-and-printed-alike",
        ),
        pytest.param(
            f"{receipt('once')}\n{block(receipt('once'))}",
            (receipt("once"),),
            id="written-and-printed-is-one-trial",
        ),
        pytest.param("epoch 1 loss 0.4\nepoch 2 loss 0.3\n", (), id="no-receipts"),
    ],
)
def test_one_harvest_reads_every_receipt_a_log_carries(log: str, expected: tuple[str, ...]):
    assert receipts_in(log) == expected


@given(before=st.lists(TEXT), after=st.lists(TEXT))
def test_the_job_s_own_output_survives_with_every_frame_line_taken_out(
    before: list[str], after: list[str]
) -> None:
    said = "".join(f"{line}\n" for line in before)
    later = "".join(f"{line}\n" for line in after)
    torn = "  mainboard-receipts-begin\nmainboard-receipt:QUJD\n"
    assert printed(f"{said}{block(_RECEIPT)}\n{torn}{later}") == said + later
    assert printed(said) == said


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
    """The only check of the coreutils pipeline itself; the rest read frames built in Python.

    An exact chunk boundary is where a frame can look right and still be wrong.
    """
    line = json.dumps({"trial_receipt": {"run_id": "r1", "outcome": "passed", "pad": "x" * pad}})
    script = f"""{staging()}
printf '%s\\n' {shlex.quote(line)} >> "${RECEIPTS_VAR}"
echo "epoch 1 loss 0.4"
{framing()}
"""
    emitted = subprocess.run(
        [posix_bash, "-c", script], capture_output=True, text=True, check=True
    ).stdout
    assert max(len(each) for each in emitted.splitlines()) < 500
    assert unframed(emitted) == f"{line}\n"
    assert receipts_in(emitted) == (line,)


@pytest.mark.parametrize("code", [0, 7], ids=["a-command-that-succeeded", "one-that-failed"])
def test_an_image_missing_the_tools_costs_its_receipts_and_never_the_jobs_exit_code(
    code: int, posix_bash: str
):
    # Emptying PATH after the command ran is exactly an image without base64, tr, fold and sed.
    script = f"""set -euo pipefail
{staging()}
printf '%s\\n' '{{"trial_receipt": {{"run_id": "r1"}}}}' >> "${RECEIPTS_VAR}"
status=0
bash -c 'exit {code}' || status=$?
PATH=/nonexistent
{framing()}
exit $status
"""
    bare = subprocess.run([posix_bash, "-c", script], capture_output=True, text=True, check=False)
    assert bare.returncode == code
    assert receipts_in(bare.stdout) == ()


def test_a_run_that_wrote_no_receipts_leaves_an_ordinary_log_exactly_as_it_was(posix_bash: str):
    script = f'{staging()}\necho "epoch 1 loss 0.4"\n{framing()}\n'
    done = subprocess.run([posix_bash, "-c", script], capture_output=True, text=True, check=True)
    assert done.stdout == "epoch 1 loss 0.4\n"
