import shlex
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard import MissionError, Project
from mainboard.center.migration.carrier import Carrier
from mainboard.engines.compile import Lockfile
from mainboard.engines.compile.generated import GeneratedFiles, Writer
from mainboard.engines.compile.pixi.lockfile import Solved

type _Step = tuple[str, str, Solved] | tuple[str, set[str]] | tuple[str]

_NAME = st.one_of(
    st.sampled_from(["default", "gpu", "tl", "a.b", "X_1"]),
    st.from_regex(r"[A-Za-z0-9][A-Za-z0-9._-]{0,5}", fullmatch=True),
)
_WORD = st.text(st.characters(max_codepoint=0xFFFF, exclude_categories=["Cs"]), max_size=8)
_LINE = st.one_of(
    st.text(st.characters(exclude_categories=["Cc", "Cs"], exclude_characters="'"), max_size=12),
    st.sampled_from(["[environments.gpu]", "lock = x", "version: 7"]),
)
_LOCK = st.builds(
    lambda lines, eol, tail: eol.join(lines) + tail,
    st.lists(_LINE, max_size=4),
    st.sampled_from(["\n", "\r\n"]),
    st.sampled_from(["", "'", "''"]),
)
_UNCARRIABLE = st.builds(
    lambda head, bad, tail: head + bad + tail,
    _LINE,
    st.sampled_from(["'''", "\r", "\x00", "\x7f"]),
    _LINE,
)
_STEP = st.one_of(
    st.tuples(
        st.just("put"), _NAME, st.builds(Solved, solved_from=_WORD, solved_by=_WORD, lock=_LOCK)
    ),
    st.tuples(
        st.just("refuse"),
        _NAME,
        st.builds(Solved, solved_from=_WORD, solved_by=_WORD, lock=_UNCARRIABLE),
    ),
    st.tuples(st.just("retain"), st.sets(_NAME)),
    st.tuples(st.just("checkout-with-crlf")),
)
_RECORDED = [
    ("put", name, Solved(solved_from=name, solved_by="fake", lock="version: 7\n"))
    for name in ("default", "gpu", "tl")
]


def _written(lock: Lockfile) -> bytes | None:
    """The lock file's bytes, None before the first write."""
    return lock.path.read_bytes() if lock.path.exists() else None


def _refused(lock: Lockfile, files: Writer, name: str, solved: Solved) -> None:
    """Put a solve the file cannot carry, and see that nothing was written."""
    kept = _written(lock)
    with pytest.raises(MissionError, match="cannot carry"):
        lock.put(files, name, solved)
    assert _written(lock) == kept


def _retained(lock: Lockfile, files: Writer, model: dict[str, Solved], declared: set[str]) -> None:
    """Retain `declared` in the lock and in its model, the flag saying whether a section went."""
    assert lock.retain(files, declared) is bool(model.keys() - declared)
    for name in model.keys() - declared:
        del model[name]


def _checked_out_with_crlf(lock: Lockfile, model: dict[str, Solved]) -> None:
    """End every line of the file in CRLF, as a checkout with `autocrlf` does, and the model's."""
    if (kept := _written(lock)) is not None:
        lock.path.write_bytes(kept.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    for name, solved in model.items():
        text = solved.lock.replace("\r\n", "\n").replace("\n", "\r\n")
        model[name] = solved.model_copy(update={"lock": text})


def _apply(lock: Lockfile, files: Writer, model: dict[str, Solved], step: _Step) -> None:
    """Run `step` on the lock and on the dict that models it, then check the file reads as it."""
    match step:
        case ("put", name, solved):
            lock.put(files, name, solved)
            model[name] = solved
        case ("refuse", name, solved):
            _refused(lock, files, name, solved)
        case ("retain", declared):
            _retained(lock, files, model, declared)
        case ("checkout-with-crlf",):
            _checked_out_with_crlf(lock, model)
    assert lock.environments() == model
    assert list(lock.environments()) == sorted(model)


@settings(max_examples=100)
@example(
    steps=[
        *_RECORDED,
        ("retain", {"default", "tl"}),
        ("retain", {"default", "tl"}),
    ]
)
@example(
    steps=[
        ("put", "gpu", Solved(solved_from="a", solved_by="b", lock="x\r\ny''")),
        ("checkout-with-crlf",),
        ("put", "default", _RECORDED[0][2]),
    ]
)
@example(steps=[*_RECORDED[:2], ("retain", {"gpu"})])
@example(
    steps=[
        ("put", "gpu", Solved(solved_from="a", solved_by="b", lock="x\n[environments.gpu]\ny\n")),
        ("put", "tl", _RECORDED[2][2]),
    ]
)
@given(steps=st.lists(_STEP, max_size=10))
def test_a_solve_is_recorded_beside_the_others_and_only_retain_drops_a_section(
    steps: Sequence[_Step],
) -> None:
    with tempfile.TemporaryDirectory() as raw:
        workspace = Path(raw)
        lock, model = Lockfile(workspace), {}
        with GeneratedFiles(directory=Project().out(workspace)).locked() as files:
            for step in steps:
                _apply(lock, files, model, step)


def test_the_lock_names_its_pypi_releases_and_its_conda_packages(locked: Path) -> None:
    lock = Lockfile(locked)

    assert lock.pins("linux-64") == {"duckdb": "2.0.0", "pytz": "2026.1"}
    assert lock.pins("osx-arm64") == {}
    assert lock.conda_names() == {"python": "python", "python-abi": "python_abi"}


def _arranged(workspace: Path, *, kept: str) -> Lockfile:
    """The workspace's solves as `kept` leaves them.

    kept: the committed lock beside a stale generated copy, that copy alone, or neither.
    """
    lock = Lockfile(workspace)
    copy = Project().out(workspace) / "envs/default/pixi.lock"
    if kept != "nothing":
        copy.parent.mkdir(parents=True)
        text = lock.environments()["default"].lock.replace("3.14.8", "3.18.0")
        copy.write_text(text, encoding="utf-8")
    if kept != "committed-and-a-stale-copy":
        lock.path.unlink()
    return lock


@pytest.mark.parametrize(
    ("kept", "platform", "interpreter"),
    [
        pytest.param(
            "committed-and-a-stale-copy",
            "linux-64",
            ">=3.14.0,<3.15",
            id="committed-beats-a-stale-copy",
        ),
        pytest.param(
            "committed-and-a-stale-copy",
            "linux-aarch64",
            ">=3.15.0,<3.16",
            id="each-platform-has-its-own-minor",
        ),
        pytest.param(
            "committed-and-a-stale-copy",
            "osx-arm64",
            ">=3.16.0rc1,<3.17",
            id="a-solved-candidate-admits-one",
        ),
        pytest.param(
            "a-copy-alone",
            "linux-64",
            ">=3.18.0,<3.19",
            id="a-workspace-with-only-generated-files",
        ),
    ],
)
def test_the_tool_runs_on_the_python_minor_the_default_environment_solved(
    locked: Path, kept: str, platform: str, interpreter: str
) -> None:
    lock = _arranged(locked, kept=kept)

    assert lock.python_request(platform) == interpreter
    assert (
        f"--python {shlex.quote(interpreter)} "
        in Carrier("host", uv="uv", python=interpreter).argv[-1]
    )


@pytest.mark.parametrize(
    ("kept", "platform"),
    [
        pytest.param(
            "committed-and-a-stale-copy", "win-64", id="a-platform-the-lock-does-not-solve"
        ),
        pytest.param("nothing", "linux-64", id="no-lock-at-all"),
    ],
)
def test_no_interpreter_is_chosen_where_the_lock_pins_none(
    locked: Path, kept: str, platform: str
) -> None:
    lock = _arranged(locked, kept=kept)

    with pytest.raises(
        MissionError, match=rf"must pin one Python for default on '{platform}'.*lock default"
    ):
        lock.python_request(platform)
