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
        pytest.param(
            'solved_from = ""\nenvs = "not-a-table"\n', id="an-envs-key-of-the-wrong-shape"
        ),
    ],
)
def test_an_unreadable_state_reads_as_stale_everywhere(
    written: str | None, tmp_path: Path
) -> None:
    """Empty is the safe direction, since the next write recomputes every digest anyway."""
    if written is not None:
        SyncState.path(tmp_path).write_text(written)
    assert SyncState.load(tmp_path) == SyncState()


# Ten examples rather than the profile's thirty, since this suite is the fast gate and the
# round trip writes a file per example. The out-of-order pair is pinned by `@example` below.
@settings(max_examples=10)
@given(envs=st.dictionaries(WORDS, WORDS, max_size=4), solved_from=st.one_of(st.just(""), WORDS))
@example(envs={"z-env": "1", "a-env": "2"}, solved_from="feedface")
def test_a_state_survives_the_file_it_renders_in_a_deterministic_order(
    envs: dict[str, str], solved_from: str, tmp_path: Path
) -> None:
    """One atomic replace has to carry every field back, and sorting keeps the file diffable."""
    state = SyncState(envs=envs, solved_from=solved_from)
    text = state.render()
    SyncState.path(tmp_path).write_text(text)
    assert SyncState.load(tmp_path) == state
    lines = text.splitlines()
    assert lines[lines.index("[envs]") + 1 :] == [
        f'{name} = "{envs[name]}"' for name in sorted(envs)
    ]
