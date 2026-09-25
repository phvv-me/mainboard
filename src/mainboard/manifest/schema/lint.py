from shlex import split

from pydantic import field_validator

from ...core.base import Declared

# The placeholder a tool's command line spells where the files it reads go.
FILES = "{files}"


class LintTool(Declared):
    """One formatter or linter `lint` runs over the files it matches, in each file's owner.

    The command runs with the owner directory as its working directory, so a checker that
    discovers its settings from the nearest project file (pyrefly, ty, ruff, vale) reads that
    owner's own. A command spelling `{files}` receives the matched files relative to the owner,
    batched so no command line outgrows what Windows accepts. A command without it checks the
    whole owner and runs once whenever any file it matches changed there, deletions included,
    which is what a type checker needs when an edit breaks a module the edit never touched.

    run: the command line, split the way a POSIX shell splits words and never handed to one,
        with `{files}` expanding to the matched files and `{root}` to the workspace root.
    files: gitignore-style patterns naming the files the tool reads, `*.py` matching at any depth.
    exclude: gitignore-style patterns this tool skips on top of `[lint].exclude`.
    writes: whether the command rewrites the files it is given, which runs it in the fix phase,
        in declaration order and before every read-only check.
    env: the environment whose tools the command runs with.
    timeout: seconds the command may take before it and its children are killed.
    """

    run: str
    files: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    writes: bool = False
    env: str = "default"
    timeout: float = 120.0

    @field_validator("run")
    @classmethod
    def splits(cls, run: str) -> str:
        """Refuse an empty command or one whose quoting never closes, at load rather than use."""
        if not split(run):
            raise ValueError("a lint tool needs a command to run")
        return run

    @property
    def argv(self) -> list[str]:
        """The command's words, placeholders still unexpanded."""
        return split(self.run)

    @property
    def per_file(self) -> bool:
        """Whether the command takes the matched files rather than checking its whole owner."""
        return FILES in self.argv


class Lint(Declared):
    """The `[lint]` table: what `lint` leaves alone, who owns what, and the tools it runs.

    Text hygiene is built in and needs no declaration: every text file `lint` reads is stored
    as UTF-8 with the newline its `.gitattributes` names (LF unless `eol=crlf`), no trailing
    whitespace and exactly one final newline, and a file carrying conflict markers or a private
    key, a TOML or YAML file that does not parse, or a file entering git above `max-kb` fails.

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
