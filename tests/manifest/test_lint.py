import pytest
from pydantic import ValidationError

from mainboard.manifest import Lint, LintTool


@pytest.mark.parametrize(
    ("check", "fix", "refusal"),
    [
        ("", "", "needs a command that checks"),
        ("ruff check 'unclosed", "", "No closing quotation"),
        ("ruff check", "ruff check --fix 'unclosed", "No closing quotation"),
    ],
    ids=["an empty check", "a check whose quoting never closes", "a fix that never closes"],
)
def test_a_tool_command_that_cannot_split_is_refused_at_load(
    check: str, fix: str, refusal: str
) -> None:
    with pytest.raises(ValidationError, match=refusal):
        LintTool(check=check, fix=fix, files=("*.py",))


def test_a_writer_fixes_unless_the_pass_is_a_check_and_a_checker_always_checks() -> None:
    writer = LintTool(check="ruff format --check {files}", fix="ruff format {files}", files=("*",))
    checker = LintTool(check="pyrefly check", fix="  ", files=("*",))

    assert writer.writes and not checker.writes
    assert checker.fix == ""
    assert writer.argv(check=False) == ["ruff", "format", "{files}"]
    assert writer.argv(check=True) == ["ruff", "format", "--check", "{files}"]
    assert checker.argv(check=False) == checker.argv(check=True) == ["pyrefly", "check"]


def test_the_table_reads_its_kebab_case_keys_and_keeps_the_declared_tool_order() -> None:
    table = Lint.model_validate(
        {
            "max-kb": 5,
            "tools": {
                "format": {
                    "check": "ruff format --check {files}",
                    "fix": "ruff format {files}",
                    "files": ["*.py"],
                },
                "check": {"check": "ruff check {files}", "files": ["*.py"]},
            },
        }
    )

    assert table.max_kb == 5
    assert list(table.tools) == ["format", "check"]
    assert table.steps == ["text", "format", "check"]
    assert table.markers == ("pyproject.toml", ".git")


def test_no_tool_may_take_the_name_of_the_built_in_hygiene() -> None:
    with pytest.raises(ValidationError, match="built-in text hygiene"):
        Lint.model_validate({"tools": {"text": {"check": "vale {files}", "files": ["*.md"]}}})
