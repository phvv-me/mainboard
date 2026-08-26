import base64
import json

import pytest

from mainboard.dispatch.evidence import RECEIPTS_VAR, framing, receipts_in, staging, unframed

# One trial receipt long enough that vast's 500-character line limit would cut it in half, which
# is the whole reason this channel exists.
_RECEIPT = json.dumps({"trial_receipt": {"run_id": "r1", "outcome": "passed", "pad": "x" * 900}})


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
