from mainboard.dispatch.state import Cache
from mainboard.dispatch.state.captured import Captured


def test_a_transcript_comes_back_byte_for_byte_and_the_newest_capture_wins() -> None:
    """A settle may capture a run twice; the later, longer tail is the one read."""
    kept = Captured(Cache.private().session)
    assert kept.transcript("s", "7") is None
    kept.keep_transcript("s", "7", "epoch 1\r\nepoch 2\n")
    assert kept.transcript("s", "7") == "epoch 1\r\nepoch 2\n"
    kept.keep_transcript("s", "7", "epoch 1\nepoch 2\nepoch 3")
    assert kept.transcript("s", "7") == "epoch 1\nepoch 2\nepoch 3"
    assert kept.transcript("s", "7", last=2) == "epoch 2\nepoch 3"
    rows = kept.session.rows("SELECT count(*) FROM lake.log_lines")
    kept.keep_transcript("s", "7", "epoch 1\nepoch 2\nepoch 3")
    assert kept.session.rows("SELECT count(*) FROM lake.log_lines") == rows


def test_a_receipt_is_kept_once_per_stream_with_the_fields_a_query_filters_on() -> None:
    kept = Captured(Cache.private().session)
    line = '{"trial_receipt": {"run": "r1", "trial": "t", "verdict": "passed", "host": "gold"}}'
    kept.keep_receipts("s", [line, "not json", line])
    kept.keep_receipts("s", [line])
    kept.keep_receipts("other", [line])
    assert kept.receipts("s") == [line, "not json"]
    rows = kept.session.rows(
        "SELECT n, run, trial, verdict, host FROM lake.receipts WHERE batch = 's' ORDER BY n"
    )
    assert rows == [(1, "r1", "t", "passed", "gold"), (2, None, None, None, None)]
