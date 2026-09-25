from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.cli import build
from mainboard.delimiter import Delimiter

from .strategies import TEXT, WORDS

# Some of the options `run` declares, each once, with a value in either spelling a caller types:
# the value as its own token or joined on with `=`.
_OPTIONS = st.lists(
    st.tuples(st.sampled_from(["--on", "--env", "--container"]), WORDS, st.booleans()),
    max_size=3,
    unique_by=lambda option: option[0],
).map(
    lambda options: [
        token
        for name, value, joined in options
        for token in ([f"{name}={value}"] if joined else [name, value])
    ]
)

# A command: a program name, then anything at all, this tool's own flags and delimiters included.
_COMMANDS = st.tuples(
    WORDS,
    st.lists(st.one_of(TEXT, st.sampled_from(["--", "-", "-h", "--help", "--on", "--version"]))),
).map(lambda command: [command[0], *command[1]])


@given(options=_OPTIONS, command=_COMMANDS)
def test_the_command_starts_at_the_first_token_that_is_not_an_option_of_the_verb(
    tmp_path_factory: pytest.TempPathFactory, options: list[str], command: list[str]
) -> None:
    """Whatever the command holds reaches it verbatim, and the verb keeps its own options.

    Placing is idempotent, since an argv that already delimits itself is left as it is.
    """
    app = build(tmp_path_factory.getbasetemp())
    placed = Delimiter(app).placed(["run", *options, *command])

    assert placed == ["run", *options, "--", *command]
    assert Delimiter(app).placed(placed) == placed
    _, bound, _ = app.parse_args(placed)
    assert bound.args == tuple(command)


@pytest.mark.parametrize(
    ("argv", "placed"),
    [
        pytest.param(["run"], ["run"], id="no-command-at-all"),
        pytest.param(
            ["run", "--on", "gold"], ["run", "--on", "gold"], id="options-and-no-command"
        ),
        pytest.param(
            ["run", "--walltim", "1", "true"],
            ["run", "--walltim", "1", "true"],
            id="an-unknown-option-left-for-the-parser-to-refuse",
        ),
        pytest.param(
            ["submit", "--on", "gold", "--yes", "--needs", "data/a", "python", "-V"],
            ["submit", "--on", "gold", "--yes", "--needs", "data/a", "--", "python", "-V"],
            id="flags-and-repeatable-options-take-what-they-declare",
        ),
        pytest.param(
            ["submit", "--no-yes", "true"],
            ["submit", "--no-yes", "--", "true"],
            id="a-negative-flag-takes-no-value",
        ),
        pytest.param(
            ["shell", "--on", "gold", "nvidia-smi", "-L"],
            ["shell", "--on", "gold", "--", "nvidia-smi", "-L"],
            id="shell",
        ),
        pytest.param(["help", "batch", "run"], ["help", "--", "batch", "run"], id="help"),
        pytest.param(
            ["proc", "timeout", "5", "pytest", "-x"],
            ["proc", "timeout", "--", "5", "pytest", "-x"],
            id="a-nested-verb-with-a-command",
        ),
        pytest.param(["lint", "src", "--check"], ["lint", "src", "--check"], id="paths"),
        pytest.param(["check", "--json"], ["check", "--json"], id="a-verb-with-no-command"),
        pytest.param(
            ["batch", "run", "spec.toml"], ["batch", "run", "spec.toml"], id="a-nested-verb"
        ),
        pytest.param(["--help"], ["--help"], id="the-root-help"),
        pytest.param([], [], id="nothing"),
    ],
)
def test_only_a_trailing_command_verb_gets_a_delimiter_and_only_before_its_command(
    tmp_path: Path, argv: list[str], placed: list[str]
) -> None:
    assert Delimiter(build(tmp_path)).placed(argv) == placed


@pytest.mark.parametrize(
    "argv",
    [
        ["lint", "src", "--check", "docs"],
        ["lint", "--check", "src", "docs"],
        ["lint", "src", "docs", "--check"],
    ],
)
def test_options_follow_the_paths_of_a_verb_that_hands_on_no_command(
    tmp_path: Path, argv: list[str]
) -> None:
    app = build(tmp_path)
    _, bound, _ = app.parse_args(Delimiter(app).placed(argv))
    assert bound.args == (Path("src"), Path("docs")) and bound.kwargs == {"check": True}
