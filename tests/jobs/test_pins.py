from pathlib import Path

import pytest

from mainboard.core.errors import MissionError
from mainboard.jobs.pins import STAGING, Pin, hub_cache, split, stage


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


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        pytest.param(
            {"HF_HUB_CACHE": "hub", "HF_HOME": "home"}, "hub", id="an-explicit-hub-cache-wins"
        ),
        pytest.param({"HF_HOME": "home"}, "home/hub", id="the-hub-inside-a-moved-home"),
        pytest.param({}, ".cache/huggingface/hub", id="the-clients-own-default"),
    ],
)
def test_the_hub_cache_is_found_where_the_hub_client_would_look(
    variables: dict[str, str], expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pin read from anywhere else would stage a file the job's own client never cached."""
    for name in ("HF_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, str(tmp_path / value))
    for home in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(home, str(tmp_path))

    assert hub_cache() == tmp_path / expected


def test_a_pin_already_staged_at_its_cached_size_is_not_copied_again(tmp_path: Path) -> None:
    """A wave of dispatches stages the same tokenizer once, not once per job."""
    cache = tmp_path / "cache"
    held = cache / "models--o--n/snapshots/r/tokenizer.json"
    held.parent.mkdir(parents=True)
    held.write_text("{}")
    root = tmp_path / "workspace"
    root.mkdir()
    (staged,) = stage(("hf://o/n@r/tokenizer.json",), root, cache)
    (root / staged).write_text("[]")

    assert stage(("hf://o/n@r/tokenizer.json",), root, cache) == (staged,)
    assert (root / staged).read_text() == "[]"
