"""Two saved Swift inventories compared, without reading source or running a build."""

import hashlib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Annotated

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ValidationError

from ....core.errors import MissionError
from ....log import logger
from ...providers.swift import SwiftToolchain
from ..report import Report
from .constants import LANGUAGES
from .models import AppleSourceFile, SourceTotal, SwiftInventory, SyntaxStatus


class SourceChange(FrozenModel):
    before: AppleSourceFile
    after: AppleSourceFile


class SwiftComparison(Report):
    baseline: Path
    current: Path
    baseline_sha256: str
    current_sha256: str
    scope: str
    baseline_toolchain: SwiftToolchain | None
    current_toolchain: SwiftToolchain | None
    totals_before: list[SourceTotal]
    totals_after: list[SourceTotal]
    added: tuple[AppleSourceFile, ...]
    removed: tuple[AppleSourceFile, ...]
    modified: tuple[SourceChange, ...]
    reclassified: tuple[SourceChange, ...]
    unchanged_sources: int
    boundary_counts_before: dict[str, int]
    boundary_counts_after: dict[str, int]
    context_issues_before: int
    context_issues_after: int
    changed_rules: tuple[str, ...]
    notes: tuple[str, ...] = (
        "Compares saved observations; it neither reads current source nor authenticates reports.",
        "Source equality uses path and SHA256, not parser findings or target membership.",
        "Reclassification is reported separately and may change category totals without deletion.",
        "Identical content at a new path is added/removed; no semantic rename is inferred.",
        "LOC/boundary changes do not prove feature parity, safe deletion or passing builds/tests.",
    )

    @classmethod
    def of(cls, baseline: Path, *, current: Path) -> SwiftComparison:
        old, old_digest = cls._read(baseline)
        new, new_digest = cls._read(current)
        cls._consistent(old, new=new)
        before = {file.path: file for file in old.files}
        after = {file.path: file for file in new.files}
        common = sorted(before.keys() & after.keys())
        old_rules = {rule.id: rule for rule in old.rules}
        new_rules = {rule.id: rule for rule in new.rules}
        return cls(
            baseline=baseline.resolve(),
            current=current.resolve(),
            baseline_sha256=old_digest,
            current_sha256=new_digest,
            scope=old.scope,
            baseline_toolchain=old.toolchain,
            current_toolchain=new.toolchain,
            totals_before=old.totals,
            totals_after=new.totals,
            added=tuple(after[name] for name in sorted(after.keys() - before.keys())),
            removed=tuple(before[name] for name in sorted(before.keys() - after.keys())),
            modified=tuple(
                SourceChange(before=before[name], after=after[name])
                for name in common
                if before[name].sha256 != after[name].sha256
            ),
            reclassified=tuple(
                SourceChange(before=before[name], after=after[name])
                for name in common
                if (before[name].language, before[name].kind)
                != (after[name].language, after[name].kind)
            ),
            unchanged_sources=sum(before[name].sha256 == after[name].sha256 for name in common),
            boundary_counts_before=dict(
                sorted(Counter(hit.rule for file in old.files for hit in file.boundaries).items())
            ),
            boundary_counts_after=dict(
                sorted(Counter(hit.rule for file in new.files for hit in file.boundaries).items())
            ),
            context_issues_before=len(old.context_issues),
            context_issues_after=len(new.context_issues),
            changed_rules=tuple(
                name
                for name in sorted(old_rules.keys() | new_rules.keys())
                if old_rules.get(name) != new_rules.get(name)
            ),
        )

    @staticmethod
    def _checked(file: AppleSourceFile, report: SwiftInventory, *, path: Path) -> None:
        """Refuse a source record `report`, read from `path`, could not truthfully hold."""
        name, scope = PurePosixPath(file.path), PurePosixPath(report.scope)
        escapes = not name.parts or name.is_absolute() or ".." in name.parts
        if escapes or name.as_posix() != file.path or not name.is_relative_to(scope):
            raise MissionError(f"source escapes or aliases inventory scope in {path}: {file.path}")
        if len(file.sha256) != 64 or set(file.sha256) - set("0123456789abcdef"):
            raise MissionError(f"invalid source SHA256 in {path}: {file.path}")
        if file.language not in LANGUAGES.values():
            raise MissionError(f"unsupported source language in {path}: {file.language}")
        SwiftComparison._covered(file, report, path=path)
        if min(file.byte_count, file.physical_lines, file.nonblank_lines) < 0 or (
            file.nonblank_lines > file.physical_lines
        ):
            raise MissionError(f"invalid source counts in {path}: {file.path}")

    @staticmethod
    def _consistent(old: SwiftInventory, *, new: SwiftInventory) -> None:
        """Refuse two inventories of different scopes, or one source measured two ways."""
        if old.scope != new.scope:
            raise MissionError(
                f"cannot compare different inventory scopes: {old.scope}, {new.scope}"
            )
        measurements: dict[str, tuple[int, int, int]] = {}
        for file in (*old.files, *new.files):
            counts = (file.byte_count, file.physical_lines, file.nonblank_lines)
            if measurements.setdefault(file.sha256, counts) != counts:
                raise MissionError(
                    f"contradictory counts for source SHA256 {file.sha256}: {file.path}"
                )

    @staticmethod
    def _covered(file: AppleSourceFile, report: SwiftInventory, *, path: Path) -> None:
        """Refuse a record whose syntax status, parse flag and parser gaps disagree."""
        expected = SwiftComparison._expected(file, report, path=path)
        unparsed = expected in (SyntaxStatus.not_swift, SyntaxStatus.excluded)
        if (
            file.syntax_status != expected
            or file.parsed != (expected == SyntaxStatus.parsed)
            or (unparsed and file.parser_gaps is not None)
        ):
            raise MissionError(f"contradictory parser coverage in {path}: {file.path}")

    @staticmethod
    def _expected(file: AppleSourceFile, report: SwiftInventory, *, path: Path) -> SyntaxStatus:
        """The syntax status `file`'s record must carry.

        A linted Swift file the parser did not cover is refused.
        """
        if file.language != "swift":
            return SyntaxStatus.not_swift
        if not file.lint_scope:
            return SyntaxStatus.excluded
        if file.parser_gaps is None or file.parser_gaps < 0 or report.toolchain is None:
            raise MissionError(f"incomplete parser coverage in {path}: {file.path}")
        return SyntaxStatus.parse_error if file.parser_gaps else SyntaxStatus.parsed

    @staticmethod
    def _read(path: Path) -> tuple[SwiftInventory, str]:
        """The inventory saved at `path`, checked record by record, and its bytes' SHA-256."""
        try:
            report = SwiftInventory.model_validate_json(data := path.read_bytes(), strict=True)
        except (OSError, ValidationError) as fault:
            raise MissionError(f"invalid Swift inventory {path}: {fault}") from fault
        SwiftComparison._scoped(report, path=path)
        for file in report.files:
            SwiftComparison._checked(file, report, path=path)
        if report.totals != SwiftInventory.totals_for(report.files):
            raise MissionError(f"inventory totals disagree with source records in {path}")
        return report, hashlib.sha256(data).hexdigest()

    @staticmethod
    def _scoped(report: SwiftInventory, *, path: Path) -> None:
        """Refuse a scope that is not a plain relative path, or a repeated source or rule."""
        scope = PurePosixPath(report.scope)
        if (
            not report.scope
            or scope.as_posix() != report.scope
            or scope.is_absolute()
            or (".." in scope.parts)
        ):
            raise MissionError(f"invalid inventory scope in {path}")
        names = [file.path for file in report.files]
        rules = {rule.id for rule in report.rules}
        if len(names) != len(set(names)) or len(report.rules) != len(rules):
            raise MissionError(f"duplicate source paths or rules in {path}")

    def text(self) -> str:
        lines = [
            f"Swift source observations: {self.baseline} -> {self.current}",
            f"{len(self.added)} added, {len(self.removed)} removed, "
            f"{len(self.modified)} modified, {len(self.reclassified)} reclassified; "
            f"{self.unchanged_sources} same-path byte matches",
        ]
        before = {(total.language, total.kind): total for total in self.totals_before}
        after = {(total.language, total.kind): total for total in self.totals_after}
        for key in sorted(before.keys() | after.keys()):
            old = before[key].physical_lines if key in before else 0
            new = after[key].physical_lines if key in after else 0
            lines.append(f"{key[0]}\t{key[1]}\t{old} -> {new} physical lines ({new - old:+d})")
        lines.extend(f"added\t{file.path}" for file in self.added)
        lines.extend(f"removed\t{file.path}" for file in self.removed)
        lines.extend(f"modified\t{change.after.path}" for change in self.modified)
        lines.extend(
            f"reclassified\t{change.after.path}: {change.before.kind} -> {change.after.kind}"
            for change in self.reclassified
        )
        if self.baseline_toolchain != self.current_toolchain or self.changed_rules:
            lines.append(
                "Parser/toolchain or rule definitions changed; review evidence provenance."
            )
        return "\n".join([*lines, *self.notes])


def swift_compare(
    baseline: Path,
    current: Path,
    *,
    as_json: Annotated[bool, Parameter(name="--json")] = False,
) -> None:
    """Compare two saved Swift inventories without Xcode, source edits or test execution.

    baseline: the earlier `swift-inventory --json` artifact.
    current: the later artifact for the same repository-relative scope.
    as_json: retain per-file changes, artifact hashes, category totals and parser provenance.
    """
    logger.result(SwiftComparison.of(baseline, current=current).render(as_json=as_json))
