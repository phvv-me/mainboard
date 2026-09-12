from pathlib import Path

import pytest

from mainboard.core.errors import MissionError
from mainboard.jobs.pins import STAGING, Pin, split, stage


def test_a_pin_is_spelled_repo_revision_and_file() -> None:
    pin = Pin.parse("hf://Qwen/Qwen3-1.7B@70d244cc/tokenizer.json")
    assert (pin.repo, pin.revision, pin.filename) == (
        "Qwen/Qwen3-1.7B",
        "70d244cc",
        "tokenizer.json",
    )
    assert pin.relative == "models--Qwen--Qwen3-1.7B/snapshots/70d244cc/tokenizer.json"
    for wrong in (
        "Qwen/Qwen3-1.7B@70d244cc/tokenizer.json",
        "hf://Qwen@70d244cc/x",
        "hf://a/b@rev",
    ):
        with pytest.raises(MissionError, match="spelled"):
            Pin.parse(wrong)


def test_needs_split_into_paths_and_pins_in_order() -> None:
    paths, pins = split(["data/a.bin", "hf://o/n@r/f", "data/b.bin"])
    assert paths == ("data/a.bin", "data/b.bin") and pins == ("hf://o/n@r/f",)


def test_staging_copies_a_cached_pin_under_the_workspace_and_names_a_missing_one(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    held = cache / "models--o--n/snapshots/r/tokenizer.json"
    held.parent.mkdir(parents=True)
    held.write_text("{}")
    root = tmp_path / "workspace"
    root.mkdir()
    staged = stage(("hf://o/n@r/tokenizer.json",), root, cache)
    assert staged == (f"{STAGING}/models--o--n/snapshots/r/tokenizer.json",)
    assert (root / staged[0]).read_text() == "{}"
    with pytest.raises(MissionError, match="hf download o/n other.json --revision r"):
        stage(("hf://o/n@r/other.json",), root, cache)
