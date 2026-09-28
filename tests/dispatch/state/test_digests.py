from typing import TYPE_CHECKING

from mainboard.dispatch.agent.program import digest
from mainboard.dispatch.state.digests import KeptDigests

if TYPE_CHECKING:
    from pathlib import Path


def rows(kept: KeptDigests) -> int:
    return int(kept.session.rows("SELECT count(*) FROM lake.digests")[0][0])


def test_only_a_moved_stamp_costs_a_row_and_a_pruned_path_is_dropped(tmp_path: Path) -> None:
    """A mirror of an unchanged tree appends nothing, and a later memory recalls every stamp."""
    source = tmp_path / "a.txt"
    source.write_text("one", encoding="utf-8")
    kept = KeptDigests(tmp_path, "mirror")
    assert kept.of(str(source), key="a.txt") == digest(str(source))
    kept.save()
    assert rows(kept) == 1

    again = KeptDigests(tmp_path, "mirror")
    assert again.held == kept.held
    again.of(str(source), key="a.txt")
    again.save()
    assert rows(again) == 1
    assert KeptDigests(tmp_path, "collection").held == {}

    source.write_text("two!", encoding="utf-8")
    again.of(str(source), key="a.txt")
    again.save()
    assert rows(again) == 2
    pruned = KeptDigests(tmp_path, "mirror")
    pruned.save(prune=True)
    assert rows(pruned) == 3
    assert KeptDigests(tmp_path, "mirror").held == {}
