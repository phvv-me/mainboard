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
    """Empty is the safe direction, since the next write recomputes every digest anyway."""
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


# Ten examples rather than the profile's thirty, since this suite is the fast gate and the
# round trip writes a file per example. The out-of-order pair is pinned by `@example` below.
@settings(max_examples=10)
@given(
    environment=WORDS,
    compiled_from=WORDS,
    solved_from=st.one_of(st.just(""), WORDS),
    solved_by=st.one_of(st.just(""), WORDS),
    compiled_at=st.one_of(st.just(""), WORDS),
    runtime_from=st.one_of(st.just(""), WORDS),
)
@example(
    environment="serving",
    compiled_from="cafebabe",
    solved_from="feedface",
    solved_by="0.77.0",
    compiled_at="/work/xg25g007/x10537/projects",
    runtime_from="deadbeef",
)
def test_a_state_survives_the_file_it_renders_in_a_deterministic_order(
    environment: str,
    compiled_from: str,
    solved_from: str,
    solved_by: str,
    compiled_at: str,
    runtime_from: str,
    tmp_path: Path,
) -> None:
    """One atomic replace carries the shard identity, its digests, solver and root back."""
    state = SyncState(
        environment=environment,
        compiled_from=compiled_from,
        solved_from=solved_from,
        solved_by=solved_by,
        compiled_at=compiled_at,
        runtime_from=runtime_from,
    )
    text = state.render()
    SyncState.path(tmp_path).write_text(text)
    assert SyncState.load(tmp_path) == state
    assert text.endswith(
        f'environment = "{environment}"\ncompiled_from = "{compiled_from}"\n'
        f'solved_from = "{solved_from}"\nsolved_by = "{solved_by}"\n'
        f'compiled_at = "{compiled_at}"\nruntime_from = "{runtime_from}"\n'
    )
