"""Recover complete receipts after dependencies emit unterminated diagnostics."""

import pytest

from mainboard.dispatch.evidence import framed, receipts_in


@pytest.mark.parametrize("legacy", (False, True))
def test_receipts_survive_unterminated_dependency_output(legacy: bool) -> None:
    receipt = '{"trial_receipt":{"run":"captured","verdict":"known"}}'
    frame = framed((receipt + "\n").encode())
    if legacy:
        frame = frame.lstrip("\n")
    log = "nanobind: shutdown diagnostic without newline" + frame
    assert receipts_in(log) == (receipt,)
    assert receipts_in(log.replace("mainboard-receipts-end", "")) == ()
