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


# Ten examples rather than the profile's thirty, since this suite is the fast gate and the
# round trip writes a file per example. The out-of-order pair is pinned by `@example` below.
@settings(max_examples=10)
@given(environment=WORDS, compiled_from=WORDS, solved_from=st.one_of(st.just(""), WORDS))
@example(environment="serving", compiled_from="cafebabe", solved_from="feedface")
def test_a_state_survives_the_file_it_renders_in_a_deterministic_order(
    environment: str, compiled_from: str, solved_from: str, tmp_path: Path
) -> None:
    """One atomic replace carries the shard identity and its two freshness digests back."""
    state = SyncState(
        environment=environment, compiled_from=compiled_from, solved_from=solved_from
    )
    text = state.render()
    SyncState.path(tmp_path).write_text(text)
    assert SyncState.load(tmp_path) == state
    assert text.endswith(
        f'environment = "{environment}"\ncompiled_from = "{compiled_from}"\n'
        f'solved_from = "{solved_from}"\n'
    )
