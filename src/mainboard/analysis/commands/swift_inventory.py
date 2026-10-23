"""Account for Apple source size and placed migration evidence without deciding deletability."""

import hashlib
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum, auto
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ValidationError

from ...core.errors import MissionError
from ..facts import Budget, Facts, lint_excludes, marked, project_files, vendored
from ..findings import Finding, Span
from ..lint import app
from ..providers.swift import SwiftSource, SwiftToolchain
from ..rules.general.apple import ObjectiveCImplementation
from ..rules.swift.migration import (
    HandwrittenDatabaseDependency,
    ObjectiveCExposure,
    PlatformFrameworkImport,
    PlatformUIBoundary,
    SwiftParserCoverage,
)
from ..rules.swift.persistence import (
    DirectSQLiteCall,
    OpaqueModelAttribute,
    PersistenceConstruction,
)
from ..rules.swift.platform import UIKitSymbolReference
from ..rules.swift.style import LegacyNavigationView, ObservableModelMigration
from ..rules.swift.ui import PrivateViewState, RedundantBodyErasure
from .catalog import RuleEntry
from .report import Report
from .scope import Scope

if TYPE_CHECKING:
    from collections.abc import Sequence

_LANGUAGES = {".swift": "swift", ".m": "objective_c", ".mm": "objective_cpp"}
_RULES = (
    SwiftParserCoverage,
    PrivateViewState,
    RedundantBodyErasure,
    LegacyNavigationView,
    ObservableModelMigration,
    PlatformFrameworkImport,
    PlatformUIBoundary,
    ObjectiveCExposure,
    HandwrittenDatabaseDependency,
    UIKitSymbolReference,
    OpaqueModelAttribute,
    PersistenceConstruction,
    DirectSQLiteCall,
)


class SourceKind(StrEnum):
    production = auto()
    test = auto()
    tooling = auto()
    generated = auto()
    vendored = auto()


class SyntaxStatus(StrEnum):
    pending = auto()
    parsed = auto()
    parse_error = auto()
    excluded = auto()
    not_swift = auto()


class Condition(Span):
    """Unevaluated #if/#elseif syntax, including branches inactive for the current platform."""

    expression: str


class AppleSourceFile(FrozenModel):
    path: str
    language: str
    kind: SourceKind
    classification: str
    lint_scope: bool
    byte_count: int
    sha256: str
    physical_lines: int
    nonblank_lines: int
    syntax_status: SyntaxStatus
    syntax_reason: str | None
    parsed: bool = False
    parser_gaps: int | None = None
    parser_message: str | None = None
    imports: tuple[str, ...] = ()
    conditions: tuple[Condition, ...] = ()
    boundaries: tuple[Finding, ...] = ()


class SourceTotal(FrozenModel):
    language: str
    kind: SourceKind
    files: int
    physical_lines: int
    nonblank_lines: int
    lint_files: int
    parsed_files: int


class SwiftInventory(Report):
    root: Path
    scope: str
    observed_at: datetime
    toolchain: SwiftToolchain | None
    context_issues: tuple[Finding, ...]
    files: list[AppleSourceFile]
    totals: list[SourceTotal]
    rules: list[RuleEntry]
    notes: tuple[str, ...] = (
        "Physical/nonblank lines include comments; they are not logical or executable LOC.",
        "Categories use git attributes and documented path/name conventions, "
        "not target membership.",
        "Generated precedence: linguist-generated, Generated directory, .pb.swift or "
        "+Generated.swift; vendor marks precede all categories. Tests precede tooling.",
        "Imports are source dependencies, not resolved packages, versions or transitive links. "
        "Conditional imports/branches are included without platform evaluation.",
        "Syntax reads the existing analysis scope ([lint].exclude and the mb tool's exclusions), "
        "including generated files if that scope "
        "keeps them. Excluded files contribute size; missing syntax never means zero boundaries.",
        "Boundary queries use catalog defaults and ignore per-rule disables/exclusions. "
        "They establish syntax only, not ownership, equivalent APIs, dead code or safe deletion.",
        "Toolchain metadata identifies the parser host compiler/SDK, not an app-target build. "
        "Rule provenance dates describe an offline guidance review, not a live docs refresh.",
        "Parse diagnostics in project siblings outside the reported directory appear in "
        "context_issues because syntax shadows use project declaration context.",
        "Cross-file shadows use top-level declarations; local names conservatively suppress "
        "within their own file. Lexical visibility and target membership are not resolved.",
        "Git ignores untracked build outputs. Outside git the existing fallback listing is used. "
        "Objective-C implementations are counted but not parsed; "
        "headers/build graphs are not read.",
    )

    @staticmethod
    def totals_for(files: Sequence[AppleSourceFile]) -> list[SourceTotal]:
        return [
            SourceTotal(
                language=language,
                kind=kind,
                files=len(group),
                physical_lines=sum(file.physical_lines for file in group),
                nonblank_lines=sum(file.nonblank_lines for file in group),
                lint_files=sum(file.lint_scope for file in group),
                parsed_files=sum(file.parsed for file in group),
            )
            for language, kind in sorted({(file.language, file.kind) for file in files})
            if (
                group := [file for file in files if (file.language, file.kind) == (language, kind)]
            )
        ]

    def text(self) -> str:
        lines = [
            f"{len(self.files)} Apple source files under {self.scope} ({self.root})",
            f"observed at {self.observed_at.isoformat()}",
        ]
        if self.toolchain:
            compiler_version = self.toolchain.compiler_version.replace("\n", "; ")
            lines.append(
                f"parser compiler: {compiler_version}; "
                f"SDK: {self.toolchain.sdk or 'not specified'}; "
                f"helper: {self.toolchain.helper_identity}"
            )
        lines.extend(
            f"{total.language}\t{total.kind}\t{total.files} files\t"
            f"{total.physical_lines} lines\t{total.nonblank_lines} nonblank\t"
            f"{total.lint_files} lint\t{total.parsed_files} parsed"
            for total in self.totals
        )
        for file in self.files:
            lines.append(
                f"{file.path}\t{file.kind}\t{file.physical_lines} lines\t"
                f"{file.nonblank_lines} nonblank\tlint={file.lint_scope}\t{file.syntax_status}\t"
                f"imports={','.join(file.imports) or '-'}"
            )
            if file.syntax_reason:
                lines.append(f"{file.path}: syntax coverage: {file.syntax_reason}")
            lines.extend(
                f"{condition.path}:{condition.start_line}:{condition.start_column}: "
                f"#if/#elseif {condition.expression}"
                for condition in file.conditions
            )
            lines.extend(
                f"{hit.span.path}:{hit.span.start_line}:{hit.span.start_column}: "
                f"{hit.rule} {hit.message}"
                for hit in file.boundaries
            )
        lines.extend(
            f"{hit.span.path}:{hit.span.start_line}:{hit.span.start_column}: "
            f"project context {hit.rule} {hit.message}"
            for hit in self.context_issues
        )
        return "\n".join([*lines, *self.notes])


class SwiftSourceInventory:
    """The source listing used by analysis, retaining excluded code for size accounting."""

    def __init__(self, scope: Scope) -> None:
        self.scope = scope

    def read(self, *, memory: str = "2GB") -> SwiftInventory:
        root = self.scope.root
        names = [name for name in project_files(root) if Path(name).suffix.lower() in _LANGUAGES]
        vendor = vendored(root, names)
        generated = marked(root, names, "linguist-generated")
        excluded = lint_excludes(root)
        files: dict[str, AppleSourceFile] = {}
        for name in sorted(names):
            if self.scope.prefix and not name.startswith(self.scope.prefix + "/"):
                continue
            path = root / name
            if path.is_symlink():
                raise MissionError(f"source inventory refuses a symbolic link: {path}")
            data = path.read_bytes()
            kind, reason = self._kind(name, vendor, generated)
            lint_scope = name not in vendor and not excluded(name)
            swift = path.suffix.lower() == ".swift"
            if not swift:
                status = SyntaxStatus.not_swift
                syntax_reason = "Objective-C file inventory only; no Objective-C parser"
            elif lint_scope:
                status, syntax_reason = SyntaxStatus.pending, None
            else:
                status = SyntaxStatus.excluded
                syntax_reason = "Excluded by vendor attribute or analysis lint scope"
            files[name] = AppleSourceFile(
                path=name,
                language=_LANGUAGES[path.suffix.lower()],
                kind=kind,
                classification=reason,
                lint_scope=lint_scope,
                byte_count=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                physical_lines=len(data.splitlines()),
                nonblank_lines=sum(bool(line.strip()) for line in data.splitlines()),
                syntax_status=status,
                syntax_reason=syntax_reason,
            )
        toolchain = None
        context_issues: tuple[Finding, ...] = ()
        if any(file.language == "swift" and file.lint_scope for file in files.values()):
            for name in names:
                if (
                    Path(name).suffix.lower() == ".swift"
                    and name not in vendor
                    and not excluded(name)
                ) and (root / name).is_symlink():
                    raise MissionError(f"Swift inventory refuses a symbolic link: {name}")
            budget = Budget(memory=memory)
            facts = Facts(root, budget)
            try:
                provider = SwiftSource()
                provider.register(facts, provider.run(root))
                toolchain, context_issues = self._syntax(facts, files)
            finally:
                facts.close()
                budget.close()
        legacy = ObjectiveCImplementation()
        for name, file in files.items():
            if file.language != "swift" and file.lint_scope:
                files[name] = file.model_copy(
                    update={
                        "boundaries": (
                            Finding(
                                rule=legacy.id,
                                span=Span(path=name, start_line=1),
                                message=legacy.message or legacy.summary(),
                            ),
                        ),
                    }
                )
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != file.sha256:
                raise MissionError(f"source changed during inventory: {name}; rerun when stable")
        return SwiftInventory(
            root=root,
            scope=self.scope.prefix or ".",
            observed_at=datetime.now(UTC),
            toolchain=toolchain,
            context_issues=context_issues,
            files=list(files.values()),
            totals=SwiftInventory.totals_for(list(files.values())),
            rules=[RuleEntry.of(rule) for rule in (*_RULES, ObjectiveCImplementation)],
        )

    @staticmethod
    def _kind(name: str, vendor: set[str], generated: set[str]) -> tuple[SourceKind, str]:
        path = PurePosixPath(name)
        parts = {part.casefold() for part in path.parts[:-1]}
        leaf = path.name.casefold()
        if name in vendor:
            return SourceKind.vendored, "git linguist-vendored attribute"
        if name in generated:
            return SourceKind.generated, "git linguist-generated attribute"
        if "generated" in parts or leaf.endswith((".pb.swift", "+generated.swift")):
            return SourceKind.generated, "Generated path or generated filename convention"
        if parts & {
            "tests",
            "test",
            "uitests",
            "apptests",
            "appstorescreenshots",
            "testsupport",
        } or leaf.endswith("tests.swift"):
            return SourceKind.test, "test directory or Tests.swift filename convention"
        if leaf == "package.swift" or parts & {"tools", "scripts", "generators", "plugins"}:
            return SourceKind.tooling, "package manifest or tooling directory convention"
        return SourceKind.production, "remaining source path; target membership is not inferred"

    @staticmethod
    def _syntax(
        facts: Facts, files: dict[str, AppleSourceFile]
    ) -> tuple[SwiftToolchain, tuple[Finding, ...]]:
        context_issues: list[Finding] = []
        for name, gaps, message in facts.db.execute(
            "SELECT path, parser_gaps, parser_message FROM swift_source_facts"
        ).fetchall():
            if name in files:
                files[name] = files[name].model_copy(
                    update={
                        "parsed": gaps == 0,
                        "parser_gaps": gaps,
                        "parser_message": message or None,
                        "syntax_status": SyntaxStatus.parse_error if gaps else SyntaxStatus.parsed,
                        "syntax_reason": (
                            "SwiftParser diagnostics prevent reliable style/migration evidence"
                            if gaps
                            else None
                        ),
                    }
                )
        for name, module in facts.db.execute(
            "SELECT DISTINCT path, name FROM swift_source_imports ORDER BY path, name"
        ).fetchall():
            if name in files:
                file = files[name]
                files[name] = file.model_copy(update={"imports": (*file.imports, module)})
        for name, line, column, expression in facts.db.execute(
            "SELECT path, start_line, start_column, condition FROM swift_source_conditions "
            "ORDER BY path, start_line, start_column"
        ).fetchall():
            if name in files:
                file = files[name]
                condition = Condition(
                    path=name, start_line=line, start_column=column, expression=expression
                )
                files[name] = file.model_copy(update={"conditions": (*file.conditions, condition)})
        for specification in _RULES:
            rule = specification()
            for name, line, column, message in facts.db.execute(
                rule.statement(), rule.rows().parameters
            ).fetchall():
                hit = Finding(
                    rule=rule.id,
                    span=Span(path=name, start_line=line, start_column=column),
                    message=message,
                )
                if name in files:
                    file = files[name]
                    files[name] = file.model_copy(update={"boundaries": (*file.boundaries, hit)})
                elif rule.id == SwiftParserCoverage.id:
                    context_issues.append(hit)
        for file in files.values():
            if file.language == "swift" and file.lint_scope and file.parser_gaps is None:
                raise MissionError(f"Swift syntax coverage is missing for {file.path}")
            files[file.path] = file.model_copy(
                update={
                    "boundaries": tuple(
                        sorted(
                            file.boundaries,
                            key=lambda hit: (
                                hit.span.start_line,
                                hit.span.start_column,
                                hit.rule,
                                hit.message,
                            ),
                        )
                    ),
                }
            )
        metadata = facts.db.execute("SELECT * FROM swift_source_toolchain")
        columns = [column[0] for column in metadata.description]
        rows = metadata.fetchall()
        if len(rows) != 1:
            raise MissionError("Swift syntax toolchain provenance is missing or ambiguous")
        toolchain = SwiftToolchain.model_validate(dict(zip(columns, rows[0], strict=True)))
        return toolchain, tuple(sorted(context_issues, key=lambda hit: hit.span.path))


@app.command
def swift_inventory(
    path: Path = Path(),
    *,
    as_json: Annotated[bool, Parameter(name="--json")] = False,
    memory: str = "2GB",
) -> None:
    """Per-file Swift/Objective-C size, imports and migration boundaries in the lint source scope.

    path: a directory to report; parsing retains the owning project's declaration context.
    as_json: include typed files/totals, placed evidence, classification limits and rule guidance.
    memory: the existing analysis engine's DuckDB memory budget.
    """
    report = SwiftSourceInventory(Scope.of(path)).read(memory=memory)
    print(report.render(as_json=as_json))


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
    def of(cls, baseline: Path, current: Path) -> SwiftComparison:
        old, old_digest = cls._read(baseline)
        new, new_digest = cls._read(current)
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
    def _read(path: Path) -> tuple[SwiftInventory, str]:
        try:
            data = path.read_bytes()
            report = SwiftInventory.model_validate_json(data, strict=True)
        except (OSError, ValidationError) as fault:
            raise MissionError(f"invalid Swift inventory {path}: {fault}") from fault
        names = [file.path for file in report.files]
        scope = PurePosixPath(report.scope)
        if (
            not report.scope
            or scope.as_posix() != report.scope
            or scope.is_absolute()
            or ".." in scope.parts
        ):
            raise MissionError(f"invalid inventory scope in {path}")
        if len(names) != len(set(names)) or len(report.rules) != len(
            {rule.id for rule in report.rules}
        ):
            raise MissionError(f"duplicate source paths or rules in {path}")
        for file in report.files:
            name = PurePosixPath(file.path)
            if (
                not name.parts
                or name.is_absolute()
                or ".." in name.parts
                or name.as_posix() != file.path
                or not name.is_relative_to(scope)
            ):
                raise MissionError(
                    f"source escapes or aliases inventory scope in {path}: {file.path}"
                )
            if len(file.sha256) != 64 or set(file.sha256) - set("0123456789abcdef"):
                raise MissionError(f"invalid source SHA256 in {path}: {file.path}")
            if file.language not in _LANGUAGES.values():
                raise MissionError(f"unsupported source language in {path}: {file.language}")
            expected = SyntaxStatus.not_swift
            if file.language == "swift":
                expected = SyntaxStatus.excluded
                if file.lint_scope:
                    if (
                        file.parser_gaps is None
                        or file.parser_gaps < 0
                        or report.toolchain is None
                    ):
                        raise MissionError(f"incomplete parser coverage in {path}: {file.path}")
                    expected = (
                        SyntaxStatus.parse_error if file.parser_gaps else SyntaxStatus.parsed
                    )
            if (
                file.syntax_status != expected
                or file.parsed != (expected == SyntaxStatus.parsed)
                or (
                    expected in (SyntaxStatus.not_swift, SyntaxStatus.excluded)
                    and file.parser_gaps is not None
                )
            ):
                raise MissionError(f"contradictory parser coverage in {path}: {file.path}")
            if (
                min(file.byte_count, file.physical_lines, file.nonblank_lines) < 0
                or file.nonblank_lines > file.physical_lines
            ):
                raise MissionError(f"invalid source counts in {path}: {file.path}")
        if report.totals != SwiftInventory.totals_for(report.files):
            raise MissionError(f"inventory totals disagree with source records in {path}")
        return report, hashlib.sha256(data).hexdigest()

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


@app.command
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
    print(SwiftComparison.of(baseline, current).render(as_json=as_json))
