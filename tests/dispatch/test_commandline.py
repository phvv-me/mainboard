import pytest

from mainboard.core.errors import MissionError
from mainboard.dispatch.commandline import joined, needs_shell, vetted


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        # A program and its arguments: several tokens, shell-quoted, so an argument that merely
        # contains shell punctuation keeps it as text rather than acting on the far side.
        (("python", "-c", "print(1)"), "python -c 'print(1)'"),
        (("python", "-c", "a; b"), "python -c 'a; b'"),
        (("pytest", "-q"), "pytest -q"),
        # A shell line quoted into one token: wrapped for a shell instead of quoted into a single
        # unrunnable word. This is the exit-127 that used to burn a rental.
        (("cd work && python train.py",), "bash -c 'cd work && python train.py'"),
        (("a | b",), "bash -c 'a | b'"),
        (("echo $HOME",), "bash -c 'echo $HOME'"),
        (("a\nb",), "bash -c 'a\nb'"),
        # A lone token with no shell syntax is still just a program name.
        (("true",), "true"),
        (("my program",), "'my program'"),
    ],
)
def test_argv_becomes_a_line_a_shell_can_actually_run(tokens: tuple[str, ...], expected: str):
    """Several tokens are argv and one token carrying shell syntax is a shell program."""
    assert joined(tokens) == expected


@pytest.mark.parametrize(
    ("token", "shell"),
    [("plain", False), ("--flag=1", False), ("a&&b", True), ("a;b", True), ("$(x)", True)],
)
def test_shell_syntax_is_what_tells_a_program_from_a_shell_line(token: str, shell: bool):
    """Only what a shell alone can act on counts, so a plain flag is never mistaken for one."""
    assert needs_shell(token) is shell


@pytest.mark.parametrize(
    ("line", "fragment"),
    [("", "the command is empty"), ("   ", "the command is empty"), ('python -c "a', "quotation")],
)
def test_a_command_no_shell_could_run_is_refused_before_anything_is_dispatched(
    line: str, fragment: str
):
    """The refusal is free here and costs a whole rental once the instance has booted."""
    with pytest.raises(MissionError, match=fragment):
        vetted(line)


def test_an_unbalanced_quote_inside_a_lone_shell_line_is_caught_before_it_is_wrapped():
    """The token is vetted before `bash -c` quoting balances it and hides the fault."""
    with pytest.raises(MissionError, match="bash -c"):
        joined(('cd work && python -c "print(1)',))


def test_a_runnable_line_comes_back_untouched():
    """Vetting refuses only what a shell would refuse, so nothing runnable is turned away."""
    assert vetted("cd work && python train.py") == "cd work && python train.py"
