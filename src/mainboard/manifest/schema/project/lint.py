from shlex import split
from typing import ClassVar

from pydantic import field_validator

from ....core.base import Declared


class LintTool(Declared):
    """One formatter or linter `lint` runs over the files it matches, in each file's owner.

    A pass that may write runs `fix` where declared and `check` elsewhere; `lint --check` runs
    only `check`. Commands run from the owner directory, so a checker finds that owner's
    settings, and are split like POSIX shell words but never handed to a shell. `{files}`
    expands to the matched files relative to the owner, batched under a command-line
    budget; without it the command checks the whole owner once whenever a matched file changed
    there, deletions included. `{root}` expands to the workspace root.

    check: reports what is wrong and changes nothing.
    fix: rewrites the files it is given, in the fix phase, in declaration order, before checks.
    files: gitignore-style patterns the tool reads, `*.py` matching at any depth.
    exclude: gitignore-style patterns skipped on top of `[lint].exclude`.
    path_when_unconfigured: a path the command takes in an owner whose own `pyproject.toml` has
        no `[tool.<name>]` table. A checker run in a directory without its configuration climbs to
        an ancestor's, so it is told where to look, while one with its configuration is given no
        path because a named path bypasses that configuration's excludes.
    only_configured: whether the tool runs solely in owners whose own `pyproject.toml` has a
        `[tool.<name>]` table, for a checker whose strictness is a per-project choice (mypy).
    serial: whether the tool's commands run one at a time instead of beside every other
        command, for one that fills the machine's memory (the analysis engine).
    timeout: seconds before the command and its children are killed.
    """

    # The placeholder a tool's command line spells where the files it reads go.
    FILES: ClassVar[str] = "{files}"

    check: str
    fix: str = ""
    files: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    path_when_unconfigured: str = ""
    only_configured: bool = False
    serial: bool = False
    env: str = "default"
    timeout: float = 120.0

    @field_validator("check")
    @classmethod
    def checks(cls, check: str) -> str:
        """Refuse, at load, an empty check or one whose quoting never closes."""
        if not split(check):
            raise ValueError("a lint tool needs a command that checks")
        return check

    @field_validator("fix")
    @classmethod
    def fixes(cls, fix: str) -> str:
        """Refuse an unclosed or `{files}`-less fix at load, and read a blank one as none.

        A fix over its whole owner could rewrite files lint leaves alone (a registration's sealed
        sources), where one given the files cannot.
        """
        if not split(fix):
            return ""
        if LintTool.FILES not in split(fix):
            raise ValueError(
                f"a fix must take {LintTool.FILES}, the files lint allows it to rewrite"
            )
        return fix

    @property
    def writes(self) -> bool:
        return bool(self.fix)

    def argv(self, *, check: bool) -> list[str]:
        """The words the tool runs, placeholders unexpanded; `check` forces the check command."""
        return split(self.check if check or not self.writes else self.fix)


class Lint(Declared):
    """The `[lint]` table: what `lint` leaves alone, who owns what, and the tools it runs.

    The built-in step `text` keeps every text file UTF-8 with its `.gitattributes` newline (LF
    unless `eol=crlf`), no trailing whitespace (Markdown hard-break spaces included) and one
    final newline, and fails case-conflicting names, broken symlinks, conflict markers, private
    keys, unparsable TOML or YAML, and untracked files above `max-kb`.

    exclude: gitignore-style patterns no step touches (vendored sources, frozen evidence), on
        top of what `.gitignore` hides.
    owners: globs of directories owning everything beneath them (`packages/*`), beside every
        directory holding one of `markers`.
    max_kb: the size above which a file git does not track yet is refused.
    """

    # The step name the built-in text hygiene reports under, which no declared tool may take.
    TEXT: ClassVar[str] = "text"

    exclude: tuple[str, ...] = ()
    owners: tuple[str, ...] = ()
    markers: tuple[str, ...] = ("pyproject.toml", ".git")
    max_kb: int = 1000
    tools: dict[str, LintTool] = {}

    @field_validator("tools")
    @classmethod
    def text_is_built_in(cls, tools: dict[str, LintTool]) -> dict[str, LintTool]:
        """Refuse a tool named `text`, which `--only` could not tell from the built-in hygiene."""
        if Lint.TEXT in tools:
            raise ValueError(
                f"{Lint.TEXT!r} is the built-in text hygiene; name the tool otherwise"
            )
        return tools

    @property
    def steps(self) -> list[str]:
        """Every step a pass can run, hygiene first, then tools in declared order."""
        return [Lint.TEXT, *self.tools]
