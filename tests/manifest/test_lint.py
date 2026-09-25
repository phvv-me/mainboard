import pytest
from pydantic import ValidationError

from mainboard.manifest import Lint, LintTool


@pytest.mark.parametrize(
    ("run", "refusal"),
    [("", "needs a command"), ("ruff check 'unclosed", "No closing quotation")],
    ids=["an empty command", "quoting that never closes"],
)
def test_a_tool_command_that_cannot_split_is_refused_at_load(run: str, refusal: str) -> None:
    with pytest.raises(ValidationError, match=refusal):
        LintTool(run=run, files=("*.py",))


@pytest.mark.parametrize(
    ("run", "per_file"),
    [("ruff check {files}", True), ("pyrefly check", False), ("vale '{files}.md'", False)],
    ids=["the files placeholder", "a whole-owner check", "a placeholder inside a word"],
)
def test_only_a_whole_files_word_makes_a_tool_read_its_matched_files(
    run: str, per_file: bool
) -> None:
    assert LintTool(run=run, files=("*",)).per_file is per_file


def test_the_table_reads_its_kebab_case_keys_and_keeps_the_declared_tool_order() -> None:
    table = Lint.model_validate(
        {
            "max-kb": 5,
            "tools": {
                "format": {"run": "ruff format {files}", "files": ["*.py"], "writes": True},
                "check": {"run": "ruff check {files}", "files": ["*.py"]},
            },
        }
    )

    assert table.max_kb == 5
    assert list(table.tools) == ["format", "check"]
    assert table.markers == ("pyproject.toml", ".git")
