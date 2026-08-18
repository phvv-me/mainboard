from typing import TYPE_CHECKING

from mainboard.engines.compile.state import SyncState

if TYPE_CHECKING:
    from pathlib import Path


def test_load_is_empty_and_stale_everywhere_when_no_file_exists(tmp_path: Path) -> None:
    state = SyncState.load(tmp_path)
    assert state == SyncState()
    assert state.envs == {}
    assert not state.solved_from


def test_load_is_empty_when_the_file_is_not_valid_toml(tmp_path: Path) -> None:
    SyncState.path(tmp_path).write_text("not [ valid toml")
    assert SyncState.load(tmp_path) == SyncState()


def test_load_falls_back_to_empty_envs_when_the_table_has_the_wrong_shape(tmp_path: Path) -> None:
    SyncState.path(tmp_path).write_text('solved_from = ""\nenvs = "not-a-table"\n')
    assert SyncState.load(tmp_path).envs == {}


def test_render_and_load_round_trip_every_field(tmp_path: Path) -> None:
    state = SyncState(envs={"default": "abc123", "serving": "def456"}, solved_from="feedface")
    SyncState.path(tmp_path).write_text(state.render())
    assert SyncState.load(tmp_path) == state


def test_render_sorts_envs_for_a_deterministic_file(tmp_path: Path) -> None:
    state = SyncState(envs={"z-env": "1", "a-env": "2"})
    text = state.render()
    assert text.index('a-env = "2"') < text.index('z-env = "1"')
