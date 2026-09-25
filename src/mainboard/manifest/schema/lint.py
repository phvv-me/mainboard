from shlex import split

from pydantic import field_validator

from ...core.base import Declared

# The placeholder a tool's command line spells where the files it reads go.
FILES = "{files}"

# The step name the built-in text hygiene reports under, which no declared tool may take.
TEXT = "text"


class LintTool(Declared):
    """One formatter or linter `lint` runs over the files it matches, in each file's owner.

    Every tool declares the read-only command that says whether its files are right, and a
    tool that can also put them right declares the command that does. A pass that may write
    runs `fix` where one is declared and `check` everywhere else, and `lint --check` runs only
    `check`, so a check never rewrites a file and every writer answers it.

    Each command runs with the owner directory as its working directory, so a checker that
    discovers its settings from the nearest project file (pyrefly, ty, ruff, vale) reads that
    owner's own. A command spelling `{files}` receives the matched files relative to the owner,
    batched so no command line outgrows what Windows accepts. A command without it checks the
    whole owner and runs once whenever any file it matches changed there, deletions included,
    which is what a type checker needs when an edit breaks a module the edit never touched.
    Both command lines are split the way a POSIX shell splits words and never handed to one,
    with `{files}` expanding to the matched files and `{root}` to the workspace root.

    check: the command that reports what is wrong and changes nothing.
    fix: the command that rewrites the files it is given, empty for a tool that only checks.
        A writer runs in the fix phase, in declaration order and before every check.
    files: gitignore-style patterns naming the files the tool reads, `*.py` matching at any depth.
    exclude: gitignore-style patterns this tool skips on top of `[lint].exclude`.
    env: the environment whose tools the command runs with.
    timeout: seconds the command may take before it and its children are killed.
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
        """Refuse an empty check or one whose quoting never closes, at load rather than use."""
        if not split(check):
            raise ValueError("a lint tool needs a command that checks")
        return check

    @field_validator("fix")
    @classmethod
    def fixes(cls, fix: str) -> str:
        """Refuse a fix whose quoting never closes and read a blank one as none, at load."""
        return fix if split(fix) else ""

    @property
    def writes(self) -> bool:
        """Whether the tool can rewrite its files, which a pass that may write lets it do."""
        return bool(self.fix)

    def argv(self, *, check: bool) -> list[str]:
        """The words the tool runs, placeholders still unexpanded.

        check: whether the pass must leave every file as it is, which selects `check` over `fix`.
        """
        return split(self.check if check or not self.writes else self.fix)


class Lint(Declared):
    """The `[lint]` table: what `lint` leaves alone, who owns what, and the tools it runs.

    Text hygiene is built in and needs no declaration: every text file `lint` reads is stored
    as UTF-8 with the newline its `.gitattributes` names (LF unless `eol=crlf`), no trailing
    whitespace and exactly one final newline, and a file carrying conflict markers or a private
    key, a TOML or YAML file that does not parse, or a file entering git above `max-kb` fails.
    It reports as the step `text`, so no tool may take that name.

    exclude: gitignore-style patterns no step reads or writes, such as vendored sources, frozen
        evidence and datasets. What `.gitignore` already hides never reaches `lint` at all.
    owners: glob patterns of workspace-relative directories that own everything beneath them,
        such as `packages/*`, beside every directory holding one of `markers`.
    markers: file names that make the directory holding them an owner.
    max_kb: the size in kilobytes above which a file git does not track yet is refused.
    tools: the formatters and linters, the writing ones run in the order declared here.
    """

    exclude: tuple[str, ...] = ()
    owners: tuple[str, ...] = ()
    markers: tuple[str, ...] = ("pyproject.toml", ".git")
    max_kb: int = 1000
    tools: dict[str, LintTool] = {}

    @field_validator("tools")
    @classmethod
    def text_is_built_in(cls, tools: dict[str, LintTool]) -> dict[str, LintTool]:
        """Refuse a tool named after the built-in hygiene, which `--only` could not tell apart."""
        if TEXT in tools:
            raise ValueError(f"{TEXT!r} is the built-in text hygiene; name the tool otherwise")
        return tools

    @property
    def steps(self) -> list[str]:
        """Every step a pass can run, the hygiene first and the tools in declared order."""
        return [TEXT, *self.tools]
