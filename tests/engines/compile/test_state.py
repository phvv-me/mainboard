from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard.engines.compile.state import SyncState

from ...strategies import WORDS


@pytest.mark.parametrize(
    "written",
    [
        pytest.param(None, id="no-file-at-all"),
        pytest.param("not [ valid toml", id="a-file-that-is-not-valid-toml"),
    ],
)
def test_an_unreadable_state_reads_as_stale_everywhere(
    written: str | None, tmp_path: Path
) -> None:
    if written is not None:
        SyncState.path(tmp_path).write_text(written)
    assert SyncState.load(tmp_path) == SyncState()


@pytest.mark.parametrize(
    "root", [r"C:\Users\vazva\mainboard-managed", 'C:\\A "quoted" path', "C:\\line\nreturn\rtab\t"]
)
def test_windows_roots_preserve_the_frozen_lock_blessing(root: str, tmp_path: Path) -> None:
    """Escaped paths cannot make a valid cross-host lock read as unblessed."""
    state = SyncState(environment="default", compiled_at=root, solved_from="sealed")
    SyncState.path(tmp_path).write_text(state.render(), encoding="utf-8")
    assert SyncState.load(tmp_path) == state


_MAYBE = st.one_of(st.just(""), WORDS)


# Ten examples rather than the profile's thirty: this is the fast gate and each writes a file.
@settings(max_examples=10)
@given(
    state=st.builds(
        SyncState,
        environment=WORDS,
        compiled_from=WORDS,
        solved_from=_MAYBE,
        solved_by=_MAYBE,
        compiled_at=_MAYBE,
        runtime_from=_MAYBE,
    )
)
@example(
    state=SyncState(
        environment="serving",
        compiled_from="cafebabe",
        solved_from="feedface",
        solved_by="0.77.0",
        compiled_at="/work/xg25g007/x10537/projects",
        runtime_from="deadbeef",
    )
)
def test_a_state_survives_the_file_it_renders_in_a_deterministic_order(
    state: SyncState, tmp_path: Path
) -> None:
    text = state.render()
    SyncState.path(tmp_path).write_text(text)
    assert SyncState.load(tmp_path) == state
    assert text.endswith(
        f'environment = "{state.environment}"\ncompiled_from = "{state.compiled_from}"\n'
        f'solved_from = "{state.solved_from}"\nsolved_by = "{state.solved_by}"\n'
        f'compiled_at = "{state.compiled_at}"\nruntime_from = "{state.runtime_from}"\n'
    )
