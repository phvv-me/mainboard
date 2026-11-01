"""What each file's findings were when last checked, so a check reports only what is new."""

import os
from collections import Counter
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from patos import Model
from pydantic import ValidationError

from ....core.project import Project
from ....git.process import Git
from ...findings import Report, Row, quoted, shape

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from collections.abc import Set as AbstractSet
    from pathlib import Path

# Prose a reflow rewraps, so a line there is no anchor: the markdown and LaTeX the engine reads.
_PROSE = frozenset({".md", ".tex"})

# A finding with its fingerprint and rule, as `Seen` counts it.
type Keyed = tuple[str, str, Row]


class _Record(Model):
    """What `Seen` keeps: each file's findings by fingerprint, and the pool of those whose file
    went away, kept until later findings match them."""

    files: dict[str, dict[str, int]] = {}
    pool: dict[str, int] = {}


class Seen:
    """The findings each file held when it was last checked, kept beside the project's state.

    A finding counts by a fingerprint that holds no path: its rule, its message's template (the
    report's, a list of holes kept as one, so 41 and 42 statements or a list grown by a name are
    one finding), and its anchor, the stripped text of its line. A finding on no line or on a
    folder is anchored by the base name, and one in prose, which a reflow rewraps, by the names
    its message quotes.

    A check counts each fingerprint over the files it reports and every path git calls changed or
    untracked, so a file or block moved by any means (an edit, `git mv`, `mv`, `refactor move`)
    brings its findings along: what one path gained, another lost. A recorded file no longer on
    disk leaves its findings in a pool, which later findings match first. Only the given files are
    reported; the other paths are recorded.

    Every check records what it saw, whether or not it reports only what is new. A session
    (`MB_REFACTOR_SESSION`, an agent's Stop hook) keeps a record of its own, started from the
    project's, so sessions editing one tree at once never take each other's findings as seen.
    """

    @staticmethod
    def _loaded(path: Path) -> _Record | None:
        """The record at `path`, None for none or one older than fingerprints."""
        try:
            return _Record.model_validate_json(path.read_bytes())
        except FileNotFoundError, ValidationError:
            return None

    @staticmethod
    def _picked(
        keyed: Mapping[str, list[Keyed]],
        grown: Mapping[str, Counter[str]],
        *,
        new: Counter[str],
        rules: AbstractSet[str],
    ) -> Report:
        """Of each fingerprint, as many findings as are `new`, from the paths it grew in.

        Paths and lines go in order, so one check always picks the same findings.
        """
        picked: dict[str, list[Row]] = {rule: [] for rule in rules}
        for path, room in sorted(grown.items()):
            for key, rule, row in sorted(keyed[path], key=lambda found: found[2]):
                if room[key] and new[key]:
                    room[key], new[key] = room[key] - 1, new[key] - 1
                    picked[rule].append(row)
        return Report(found=picked)

    def __init__(self, root: Path) -> None:
        self.root = root
        folder = Project().out(root) / "refactor"
        session = Project().variable("REFACTOR_SESSION").read()
        self.path = folder / "sessions" / f"{session}.json" if session else folder / "seen.json"
        self.record = Seen._loaded(self.path) or Seen._loaded(folder / "seen.json") or _Record()
        self.lines: dict[str, list[str] | None] = {}

    def _anchor(self, row: Row) -> str:
        """What places the finding `row` in its file, its path aside."""
        path, line, _, message = row
        if PurePosixPath(path).suffix in _PROSE:
            return "|".join(quoted(message))
        if path not in self.lines:
            try:
                self.lines[path] = (
                    (self.root / path).read_text(encoding="utf-8", errors="replace").splitlines()
                )
            except OSError:
                self.lines[path] = None
        lines = self.lines[path]
        if lines is None or line < 1:
            return PurePosixPath(path).name
        return lines[line - 1].strip() if line <= len(lines) else ""

    def _changed(self) -> set[str]:
        """The paths under the root git calls changed since HEAD or untracked; none outside git."""
        git = Git(self.root)
        listings = (
            git.run("diff", "HEAD", "--name-only", "--relative", "-z"),
            git.run("-c", "core.excludesFile=/dev/null", "ls-files", "-oz", "--exclude-standard"),
        )
        return {
            name
            for listing in listings
            if not listing.returncode
            for name in listing.stdout.split("\0")
            if name
        }

    def _held(self, path: str, ran: AbstractSet[str]) -> Counter[str]:
        """What `path` held when last recorded, of the rules in `ran`."""
        held = self.record.files.get(path, {})
        return Counter({key: count for key, count in held.items() if _rule(key) in ran})

    def _keyed(self, report: Report, paths: AbstractSet[str]) -> dict[str, list[Keyed]]:
        """The findings in `paths`, each with its fingerprint (rule, template, anchor), by path."""
        keyed: dict[str, list[Keyed]] = {path: [] for path in paths}
        for rule, rows in report.found.items():
            for row in rows:
                if row[0] in paths:
                    keyed[row[0]].append(
                        ("\t".join((rule, shape(row[3]), self._anchor(row))), rule, row)
                    )
        return keyed

    def _recorded(self, now: Mapping[str, Counter[str]], *, ran: AbstractSet[str]) -> None:
        """Record what each path holds of the rules in `ran`, keeping the other rules' counts."""
        for path, counts in now.items():
            held = self.record.files.get(path, {})
            self.record.files[path] = {
                key: count for key, count in held.items() if _rule(key) not in ran
            } | counts
        self.path.parent.mkdir(parents=True, exist_ok=True)
        staged = self.path.with_suffix(f".{os.getpid()}.tmp")
        staged.write_text(self.record.model_dump_json(), encoding="utf-8")
        os.replace(staged, self.path)

    def _unmatched(
        self,
        now: Mapping[str, Counter[str]],
        *,
        was: Mapping[str, Counter[str]],
        gone: AbstractSet[str],
    ) -> Counter[str]:
        """How far each fingerprint grew over the paths `now` counts, beyond what the pool held.

        The findings of the `gone` paths join the pool and leave the record; the growth is then
        drawn from the pool, which keeps what nothing matched.
        """
        pool = Counter(self.record.pool)
        for path in gone:
            pool.update(self.record.files.pop(path))
        net = sum(now.values(), Counter())
        net.subtract(sum(was.values(), Counter()))
        grown = +net
        drawn = grown & pool
        self.record.pool = dict(pool - drawn)
        return grown - drawn

    def fresh(self, report: Report, files: Collection[str]) -> Report:
        """The findings of `files` beyond what was recorded, recording this check.

        The first check of a project records without reporting. Only the rules `report` ran are
        recorded, so a check of a few rules leaves what the others last saw.
        """
        ran, first = set(report.found), not (self.record.files or self.record.pool)
        gone = {path for path in self.record.files if not os.path.lexists(self.root / path)}
        counted = {*self.record.files, *(row[0] for rows in report.found.values() for row in rows)}
        covered = {*files, *(self._changed() & counted)} - gone
        keyed = self._keyed(report, covered)
        now = {path: Counter(key for key, _, _ in found) for path, found in keyed.items()}
        was = {path: self._held(path, ran) for path in covered}
        new = self._unmatched(now, was=was, gone=gone)
        self._recorded(now, ran=ran)
        return Seen._picked(
            keyed,
            {path: now[path] - was[path] for path in (() if first else {*files} - gone)},
            new=new,
            rules=ran,
        )


def _rule(key: str) -> str:
    """The rule of the fingerprint `key`."""
    return key.partition("\t")[0]
