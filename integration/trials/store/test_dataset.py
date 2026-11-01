from pathlib import Path

from mainboard.state import records
from mainboard.trials import Admissibility, Cell, Dataset, Outcome, Probed

_AXIS = "ca\"rd%s'x"


def test_an_axis_spelled_with_quotes_and_a_percent_still_groups_and_filters(
    tmp_path: Path,
) -> None:
    dataset = Dataset(tmp_path / "receipts", axes=(_AXIS,))
    dataset.writer("20260101T000000Z-a", {"opened_at_ns": 1}).write(
        {
            "lane": "speed",
            "key": "a",
            _AXIS: "h100",
            f"{_AXIS}_probed": str(Probed.FOUND),
            "outcome": Outcome.PASSED,
            "admissibility": Admissibility.ADMISSIBLE,
        }
    )
    here = Cell(values={_AXIS: "h100"}, probing={_AXIS: Probed.FOUND})

    status = dataset.coverage.status("speed", ["a", "b"], here)

    assert [row["key"] for row in records(dataset.passing())] == ["a"]
    assert (status.have, status.missing) == (1, ("b",))
