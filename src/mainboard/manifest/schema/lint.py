from shlex import split

from pydantic import field_validator

from ...core.base import Declared

# The placeholder a tool's command line spells where the files it reads go.
FILES = "{files}"

# The step name the built-in text hygiene reports under, which no declared tool may take.
TEXT = "text"


class LintTool(Declared):
    """One formatter or linter `lint` runs over the files it matches, in each file's owner.

    A pass that may write runs `fix` where declared and `check` elsewhere; `lint --check` runs
    only `check`. Commands run from the owner directory, so a checker finds that owner's
    settings, and are split like POSIX shell words but never handed to a shell. `{files}`
    expands to the matched files relative to the owner, batched under Windows' command-line
    limit; without it the command checks the whole owner once whenever a matched file changed
    there, deletions included. `{root}` expands to the workspace root.

    check: reports what is wrong and changes nothing.
    fix: rewrites the files it is given, in the fix phase, in declaration order, before checks.
    files: gitignore-style patterns the tool reads, `*.py` matching at any depth.
    exclude: gitignore-style patterns skipped on top of `[lint].exclude`.
    timeout: seconds before the command and its children are killed.
    """

    check: str
    fix: str = ""
    files: tuple[str, ...]
    exclude: tuple[str, ...] = ()
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
        """Refuse, at load, a fix whose quoting never closes, and read a blank one as none."""
        return fix if split(fix) else ""

    @property
    def writes(self) -> bool:
        return bool(self.fix)

    def argv(self, *, check: bool) -> list[str]:
        """The words the tool runs, placeholders unexpanded; `check` forces the check command."""
        return split(self.check if check or not self.writes else self.fix)


class Lint(Declared):
    """The `[lint]` table: what `lint` leaves alone, who owns what, and the tools it runs.

    The built-in step `text` keeps every text file UTF-8 with its `.gitattributes` newline (LF
    unless `eol=crlf`), no trailing whitespace and one final newline, and fails conflict
    markers, private keys, unparsable TOML or YAML, and untracked files above `max-kb`.

    exclude: gitignore-style patterns no step touches (vendored sources, frozen evidence), on
        top of what `.gitignore` hides.
    owners: globs of directories owning everything beneath them (`packages/*`), beside every
        directory holding one of `markers`.
    max_kb: the size above which a file git does not track yet is refused.
    """

    exclude: tuple[str, ...] = ()
    owners: tuple[str, ...] = ()
    markers: tuple[str, ...] = ("pyproject.toml", ".git")
    max_kb: int = 1000
    tools: dict[str, LintTool] = {}

    @field_validator("tools")
    @classmethod
    def text_is_built_in(cls, tools: dict[str, LintTool]) -> dict[str, LintTool]:
        """Refuse a tool named `text`, which `--only` could not tell from the built-in hygiene."""
        if TEXT in tools:
            raise ValueError(f"{TEXT!r} is the built-in text hygiene; name the tool otherwise")
        return tools

    @property
    def steps(self) -> list[str]:
        """Every step a pass can run, hygiene first, then tools in declared order."""
        return [TEXT, *self.tools]
