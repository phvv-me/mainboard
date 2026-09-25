import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from pathspec import GitIgnoreSpec
from plumbum import local

from ..core.errors import MissionError
from ..core.project import Project
from ..engines.compile.provisioner import Provisioner
from ..manifest.schema.lint import FILES, TEXT
from . import text
from .inventory import Attributes, Inventory
from .owners import Owners
from .process import Invocation, Outcome
from .report import Report

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from ..manifest.schema.root import Manifest

# Characters of file arguments one command carries. Windows refuses a command line past 32767
# characters, and the same budget everywhere keeps a run's batching identical on every machine.
_BATCH = 24_000


class Linter:
    """Normalize, fix and check workspace files in one parallel pass.

    The pass has three phases, and each one only starts once the one before it has settled:
    the built-in text hygiene over every file at once, then each writing tool in the order
    `[lint.tools]` declares them, since two formatters must never rewrite one file together,
    then every read-only check at once. Each tool runs once per owner of the files it matches,
    inside that owner, under its environment's PATH.

    A check pass writes nothing: the hygiene names what it would repair as findings of its
    own, and every tool runs its `check` command, all at once since no two of them can race
    on a file.

    manifest: the workspace manifest, whose `[lint]` table drives the pass.
    check: leave every file as it is and report what a writing pass would change.
    only: the steps to run, `text` naming the hygiene, every declared step when empty.
    """

    def __init__(
        self, root: Path, manifest: Manifest, *, check: bool = False, only: Sequence[str] = ()
    ) -> None:
        self.root = root
        self.table = manifest.lint
        self.check = check
        self.steps = self._selected(only)
        self._tools = {name: tool for name, tool in self.table.tools.items() if name in self.steps}
        self.inventory = Inventory(root)
        self.owners = Owners(root, self.table.owners, self.table.markers)
        self._provisioner = Provisioner(root, manifest)
        self._environments: dict[str, Mapping[str, str]] = {}
        # The generated tree is never the workspace's own text, whether or not a `.gitignore`
        # already says so.
        self._excluded = GitIgnoreSpec.from_lines([f"/{Project().out_dir}/", *self.table.exclude])

    def lint(self, paths: Sequence[Path]) -> Report:
        """Repair text, run the writing tools in declared order, then every check at once.

        paths: absolute files beneath the root. One that no longer exists still wakes the
            checks that read its whole owner, since a deletion can break what imported it.
        """
        files = [path for path in paths if not self._excluded.match_file(self._relative(path))]
        before = {path: _fingerprint(path) for path in files}
        outcomes = self._hygiene(files) if TEXT in self.steps else []
        writers = [] if self.check else [name for name, tool in self._tools.items() if tool.writes]
        for name in writers:
            outcomes.extend(self._parallel(self._invocations(name, files)))
        checks = [
            invocation
            for name in self._tools
            if name not in writers
            for invocation in self._invocations(name, files)
        ]
        outcomes.extend(self._parallel(checks))
        return Report(
            files=len(files),
            rewritten=tuple(
                self._relative(path) for path in files if _fingerprint(path) != before[path]
            ),
            failures=tuple(outcome for outcome in outcomes if outcome.failed),
        )

    def _selected(self, only: Sequence[str]) -> list[str]:
        """The steps `only` names, every step when it names none, refusing an undeclared one."""
        if unknown := sorted(set(only) - set(self.table.steps)):
            raise MissionError(
                f"no lint step {unknown[0]!r}; the steps are {', '.join(self.table.steps)}"
            )
        return [step for step in self.table.steps if not only or step in only]

    def _hygiene(self, files: Sequence[Path]) -> list[Outcome]:
        """The built-in text repairs over every existing file, one outcome if anything is left."""
        started = time.monotonic()
        existing = [path for path in files if path.is_file()]
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
        return [Outcome(step=TEXT, owner=".", code=1, seconds=seconds, output="\n".join(lines))]

    def _examined(self, path: Path, attributes: Attributes) -> list[str]:
        """Repair one file's text, or name the repair in a check, then say what is left.

        A new file past the size limit is left too, since no repair can shrink it.
        """
        examination = text.examined(path, attributes)
        problems = list(examination.problems)
        if examination.repaired is not None:
            if self.check:
                problems.insert(0, f"needs repair: {', '.join(examination.untidy)}")
            else:
                path.write_bytes(examination.repaired)
        size = path.stat().st_size
        if size > self.table.max_kb * 1024 and not self.inventory.tracked(path):
            problems.append(f"is {size // 1024} KB, above the {self.table.max_kb} KB limit")
        return problems

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
                owned.setdefault(self.owners.of(path.parent), []).append(path)
        return [
            Invocation(
                step=name,
                owner=self._relative(owner),
                cwd=owner,
                argv=tuple(self._expanded(argv, batch)),
                env=tool.env,
                timeout=tool.timeout,
            )
            for owner, matched in sorted(owned.items())
            for batch in self._batches(argv, owner, matched)
        ]

    def _batches(
        self, argv: Sequence[str], owner: Path, matched: Sequence[Path]
    ) -> list[list[str]]:
        """The file arguments of each command `argv` runs in `owner`.

        A whole-owner check gets one empty batch, and a per-file tool gets none once every file
        it matched there is gone.
        """
        if FILES not in argv:
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

    def _expanded(self, argv: Sequence[str], files: Sequence[str]) -> Iterator[str]:
        for word in argv:
            if word == FILES:
                yield from files
            else:
                yield word.replace("{root}", str(self.root))

    def _parallel(self, invocations: Sequence[Invocation]) -> list[Outcome]:
        """Run `invocations` at once, each environment activated once for the whole pass."""
        for env in {invocation.env for invocation in invocations} - self._environments.keys():
            with self._provisioner.activated(env):
                self._environments[env] = local.env.getdict()
        with ThreadPoolExecutor() as pool:
            return list(pool.map(lambda job: job.run(self._environments[job.env]), invocations))

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()


def _fingerprint(path: Path) -> bytes | None:
    """A digest of what `path` holds, None once it is gone."""
    try:
        return hashlib.blake2b(path.read_bytes(), digest_size=16).digest()
    except FileNotFoundError:
        return None
