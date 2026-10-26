import hashlib
import re
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import cache, cached_property
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from pathspec import GitIgnoreSpec
from plumbum import local

from ..constants import BYTES_PER_KIBIBYTE
from ..core.errors import MissionError
from ..core.project import Project
from ..engines.compile.provisioner import Provisioner
from ..manifest.schema.project import lint as lint_module
from ..proc import Exit
from .files import text
from .files.inventory import Attributes, Inventory
from .owners import Owners
from .process import Invocation, Outcome
from .report import Report
from .sealed import Sealed

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence, Set
    from pathlib import Path

    from ..manifest.schema.root import Manifest

# Characters of file arguments one command carries, well inside every kernel's argument limit,
# and the same budget everywhere keeps a run's batching identical on every machine.
_BATCH = 24_000

# Where a finding line points, `path:line` as every check prints it (ruff, ty's concise and
# pyrefly's min-text formats, mcmr's concise one, vulture, vale): relative to the owner the
# check ran in, or absolute.
_LOCATION = re.compile(r"(?:^|\s)(?P<path>[^\s:]+\.\w+):\d+")


class Linter:
    """Normalize, fix and check workspace files in one parallel pass.

    The pass has three phases, and each one only starts once the one before it has settled:
    the built-in text hygiene over every file at once, then each writing tool in the order
    `[lint.tools]` declares them, since two formatters must never rewrite one file together,
    then every read-only check at once. Each tool runs once per owner of the files it matches,
    inside that owner, under its environment's PATH.

    A check that reads its whole owner (a type checker, mcmr) answers for every file there, so
    its verdict is narrowed to the files this pass reads: a finding naming another file is
    dropped, and the check passes when none is left. Output naming no file at all is a tool
    that broke, and keeps its failure. A pass over a few files is red only for those files.

    A check pass writes nothing: the hygiene names what it would repair as findings of its
    own, and every tool runs its `check` command, all at once since no two of them can race
    on a file.

    manifest: the workspace manifest, whose `[lint]` table drives the pass.
    check: leave every file as it is and report what a writing pass would change.
    only: the steps to run, `text` naming the hygiene, every declared step when empty.
    """

    @cached_property
    def _excluded(self) -> GitIgnoreSpec:
        """The generated trees, never the workspace's own text, and the `[lint] exclude` globs."""
        return GitIgnoreSpec.from_lines(
            [*(f"/{name}/" for name in Project().out_dirs), *self.table.exclude]
        )

    @cached_property
    def _tools(self) -> dict[str, lint_module.LintTool]:
        """The declared tools of the selected steps, in declaration order."""
        return {name: tool for name, tool in self.table.tools.items() if name in self.steps}

    def __init__(
        self, root: Path, manifest: Manifest, *, check: bool = False, only: Sequence[str] = ()
    ) -> None:
        self.root = root
        self.table = manifest.lint
        self.check = check
        self.steps = self._selected(only)
        self.inventory = Inventory(root, manifest.git)
        self.owners = Owners(root, self.table.owners, markers=self.table.markers)
        self.sealed = Sealed(root)
        self._configured = cache(self._configures)
        self._provisioner = Provisioner(root, manifest)
        self._environments: dict[str, Mapping[str, str]] = {}

    def _batches(
        self, argv: Sequence[str], owner: Path, matched: Sequence[Path]
    ) -> list[list[str]]:
        """The file arguments of each command `argv` runs in `owner`.

        A whole-owner check gets one empty batch, and a per-file tool gets none once every file
        it matched there is gone.
        """
        if lint_module.LintTool.FILES not in argv:
            return [[]]
        batches: list[list[str]] = []
        width = _BATCH
        for path in matched:
            if not path.is_file():
                continue
            name = path.relative_to(owner).as_posix()
            if width + len(name) >= _BATCH:
                batches.append([])
                width = 0
            batches[-1].append(name)
            width += len(name) + 1
        return batches

    def _configures(self, owner: Path, name: str) -> bool:
        """Whether `owner`'s own `pyproject.toml` has a `[tool.<name>]` table."""
        try:
            document = tomllib.loads((owner / "pyproject.toml").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False
        return name in document.get("tool", {})

    def _examined(self, path: Path, attributes: Attributes) -> list[str]:
        """Repair one file's text, or name the repair in a check, then say what is left.

        A new file past the size limit is left too, since no repair can shrink it, and so is a
        name that differs from a neighbor's only in case or Unicode form, which no checkout on a
        case-insensitive disk can hold.
        """
        examination = text.examined(path, attributes)
        problems = [*self._repairs(path, examination), *examination.problems]
        if twins := text.twins(path.name, self.inventory.siblings(path)):
            problems.append(f"differs only in case or Unicode form from {', '.join(twins)}")
        return [*problems, *self._oversize(path)]

    def _expanded(self, argv: Sequence[str], *, files: Sequence[str]) -> Iterator[str]:
        for word in argv:
            if word == lint_module.LintTool.FILES:
                yield from files
            else:
                yield word.replace("{root}", str(self.root))

    def _hygiene(self, files: Sequence[Path]) -> list[Outcome]:
        """The built-in text repairs over every existing file, one outcome if anything is left."""
        started = time.monotonic()
        existing = [path for path in files if path.is_file() or path.is_symlink()]
        attributes = self.inventory.attributes(existing)
        with ThreadPoolExecutor() as pool:
            found = pool.map(
                lambda path: self._examined(path, attributes.get(path, Attributes())), existing
            )
            lines = [
                f"{self._relative(path)}: {problem}"
                for path, problems in zip(existing, found, strict=True)
                for problem in problems
            ]
        if not lines:
            return []
        seconds = time.monotonic() - started
        return [
            Outcome(
                step=lint_module.Lint.TEXT,
                owner=".",
                code=1,
                seconds=seconds,
                output="\n".join(lines),
            )
        ]

    def _invocations(self, name: str, files: Sequence[Path]) -> list[Invocation]:
        """The commands tool `name` runs over `files`, one or more per owner of what it matches."""
        tool = self._tools[name]
        argv = tool.argv(check=self.check)
        chosen = GitIgnoreSpec.from_lines(tool.files)
        skipped = GitIgnoreSpec.from_lines(tool.exclude)
        owned: dict[Path, list[Path]] = {}
        for path in files:
            relative = self._relative(path)
            if chosen.match_file(relative) and not skipped.match_file(relative):
                owner = self.owners.owner_of(path.parent)
                if not tool.only_configured or self._configured(owner, name):
                    owned.setdefault(owner, []).append(path)
        return [
            Invocation(
                step=name,
                owner=self._relative(owner),
                cwd=owner,
                argv=tuple(self._expanded(self._scoped(name, argv, owner), files=batch)),
                env=tool.env,
                timeout=tool.timeout,
            )
            for owner, matched in sorted(owned.items())
            for batch in self._batches(argv, owner, matched)
        ]

    def _named(self, line: str, *, owner: str) -> str | None:
        """The workspace-relative file a finding line names.

        It is None for a line naming none and empty for one naming a file outside the workspace.
        """
        found = _LOCATION.search(line)
        if found is None:
            return None
        place = PurePosixPath(found["path"])
        if not place.is_absolute():
            return (PurePosixPath(owner) / place).as_posix()
        return place.relative_to(self.root).as_posix() if place.is_relative_to(self.root) else ""

    def _narrowed(self, outcome: Outcome, read: Set[str]) -> Outcome:
        """A whole-owner check's verdict over the workspace-relative files `read` alone.

        A check that timed out or never started keeps its failure whatever it printed.
        """
        if outcome.code in {0, Exit.TIMED_OUT, Exit.MISSING}:
            return outcome
        named = [
            (line, self._named(line, owner=outcome.owner)) for line in outcome.output.splitlines()
        ]
        if not any(name for _, name in named):
            return outcome
        kept = [line for line, name in named if name in read]
        code = outcome.code if kept else 0
        return outcome.model_copy(update={"code": code, "output": "\n".join(kept)})

    def _oversize(self, path: Path) -> list[str]:
        """Why `path` is past the size limit, empty when it fits or git tracks it."""
        size = path.stat().st_size if path.exists() else 0
        if size <= self.table.max_kb * BYTES_PER_KIBIBYTE or self.inventory.tracked(path):
            return []
        return [f"is {size // BYTES_PER_KIBIBYTE} KB, above the {self.table.max_kb} KB limit"]

    def _parallel(self, invocations: Sequence[Invocation]) -> list[Outcome]:
        """Run `invocations` at once, each environment activated once for the whole pass.

        The commands of a `serial` tool queue behind one another, beside the rest, so a few
        owners at once never put several memory-hungry commands in the machine together.
        """
        for env in {invocation.env for invocation in invocations} - self._environments.keys():
            with self._provisioner.activated(env):
                self._environments[env] = local.env.getdict()

        def run(job: Invocation) -> Outcome:
            return job.run(self._environments[job.env])

        heavy = [job for job in invocations if self._tools[job.step].serial]
        light = [job for job in invocations if job not in heavy]
        with ThreadPoolExecutor() as pool:
            queue = pool.submit(lambda: [run(job) for job in heavy])
            return [*pool.map(run, light), *queue.result()]

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _repairs(self, path: Path, examination: text.Examination) -> list[str]:
        """Write `path`'s repaired bytes, or name the repair as a problem when checking."""
        if examination.repaired is None:
            return []
        if self.check:
            return [f"needs repair: {', '.join(examination.untidy)}"]
        path.write_bytes(examination.repaired)
        return []

    def _scoped(self, name: str, argv: Sequence[str], owner: Path) -> list[str]:
        """`argv` ending in tool `name`'s `path_when_unconfigured`, where an owner needs it.

        An owner needs it when it has no `[tool.<name>]` table of its own.
        """
        path = self._tools[name].path_when_unconfigured
        return [*argv, path] if path and not self._configured(owner, name) else list(argv)

    def _selected(self, only: Sequence[str]) -> list[str]:
        """The steps `only` names, every step when it names none, refusing an undeclared one."""
        if unknown := sorted(set(only) - set(self.table.steps)):
            raise MissionError(
                f"no lint step {unknown[0]!r}; the steps are {', '.join(self.table.steps)}"
            )
        return [step for step in self.table.steps if not only or step in only]

    def lint(self, paths: Sequence[Path]) -> Report:
        """Repair text, run the writing tools in declared order, then every check at once.

        paths: absolute files beneath the root. One that no longer exists still wakes the
            checks that read its whole owner, since a deletion can break what imported it. A
            file a registration has sealed is left out of every step, since the digest it was
            registered under covers its bytes.
        """
        files = [
            path
            for path in paths
            if not self._excluded.match_file(self._relative(path)) and not self.sealed.holds(path)
        ]
        before = {path: _fingerprint(path) for path in files}
        outcomes = self._hygiene(files) if lint_module.Lint.TEXT in self.steps else []
        writers = [] if self.check else [name for name, tool in self._tools.items() if tool.writes]
        for name in writers:
            outcomes.extend(self._parallel(self._invocations(name, files)))
        checks = [
            invocation
            for name in self._tools
            if name not in writers
            for invocation in self._invocations(name, files)
        ]
        read = frozenset(self._relative(path) for path in files)
        whole = {
            name
            for name, tool in self._tools.items()
            if lint_module.LintTool.FILES not in tool.argv(check=self.check)
        }
        outcomes.extend(
            self._narrowed(outcome, read) if outcome.step in whole else outcome
            for outcome in self._parallel(checks)
        )
        return Report(
            files=len(files),
            rewritten=tuple(
                self._relative(path) for path in files if _fingerprint(path) != before[path]
            ),
            failures=tuple(outcome for outcome in outcomes if outcome.failed),
        )


def _fingerprint(path: Path) -> bytes | None:
    """A digest of what `path` holds, None once it is gone."""
    with suppress(FileNotFoundError):
        return hashlib.blake2b(path.read_bytes(), digest_size=16).digest()
    return None
