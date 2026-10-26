"""What a Swift inventory records: each Apple source file, its totals, and the report."""

from datetime import datetime
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from ...findings import Finding, Span
from ...providers.swift import SwiftToolchain
from ..catalog import RuleEntry
from ..report import Report

if TYPE_CHECKING:
    from collections.abc import Sequence


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
