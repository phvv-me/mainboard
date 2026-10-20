"""Account for Apple source size and placed migration evidence without deciding deletability."""

import hashlib
from enum import StrEnum, auto
from pathlib import Path, PurePosixPath
from typing import Annotated

from cyclopts import Parameter
from patos import FrozenModel

from ...core.errors import MissionError
from ..facts import Budget, Facts, lint_excludes, marked, project_files, vendored
from ..findings import Finding, Span
from ..lint import app
from ..providers.swift import SwiftSource
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
from .catalog import RuleEntry
from .report import Report
from .scope import Scope

_LANGUAGES = {".swift": "swift", ".m": "objective_c", ".mm": "objective_cpp"}
_RULES = (
    SwiftParserCoverage,
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
        "Git ignores untracked build outputs. Outside git the existing fallback listing is used. "
        "Objective-C implementations are counted but not parsed; "
        "headers/build graphs are not read.",
    )

    def text(self) -> str:
        lines = [f"{len(self.files)} Apple source files under {self.scope} ({self.root})"]
        lines.extend(
            f"{total.language}\t{total.kind}\t{total.files} files\t"
            f"{total.physical_lines} lines\t{total.nonblank_lines} nonblank\t"
            f"{total.lint_files} lint\t{total.parsed_files} parsed"
            for total in self.totals
        )
        for file in self.files:
            syntax = (
                "parsed" if file.parsed else "parse-error" if file.parser_gaps else "not-parsed"
            )
            lines.append(
                f"{file.path}\t{file.kind}\t{file.physical_lines} lines\t"
                f"{file.nonblank_lines} nonblank\tlint={file.lint_scope}\t{syntax}\t"
                f"imports={','.join(file.imports) or '-'}"
            )
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
            files[name] = AppleSourceFile(
                path=name,
                language=_LANGUAGES[path.suffix.lower()],
                kind=kind,
                classification=reason,
                lint_scope=name not in vendor and not excluded(name),
                byte_count=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                physical_lines=len(data.splitlines()),
                nonblank_lines=sum(bool(line.strip()) for line in data.splitlines()),
            )
        if any(file.language == "swift" and file.lint_scope for file in files.values()):
            for name in names:
                if (
                    Path(name).suffix.lower() == ".swift"
                    and name not in vendor and not excluded(name)
                ):
                    if (root / name).is_symlink():
                        raise MissionError(f"Swift inventory refuses a symbolic link: {name}")
            budget = Budget(memory=memory)
            facts = Facts(root, budget)
            try:
                provider = SwiftSource()
                provider.register(facts, provider.run(root))
                self._syntax(facts, files)
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
        totals = [
            SourceTotal(
                language=language,
                kind=kind,
                files=len(group),
                physical_lines=sum(file.physical_lines for file in group),
                nonblank_lines=sum(file.nonblank_lines for file in group),
                lint_files=sum(file.lint_scope for file in group),
                parsed_files=sum(file.parsed for file in group),
            )
            for language, kind in sorted({(file.language, file.kind) for file in files.values()})
            if (
                group := [
                    file for file in files.values()
                    if (file.language, file.kind) == (language, kind)
                ]
            )
        ]
        return SwiftInventory(
            root=root,
            scope=self.scope.prefix or ".",
            files=list(files.values()),
            totals=totals,
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
        if (
            parts & {"tests", "test", "uitests", "apptests", "appstorescreenshots", "testsupport"}
            or leaf.endswith("tests.swift")
        ):
            return SourceKind.test, "test directory or Tests.swift filename convention"
        if leaf == "package.swift" or parts & {"tools", "scripts", "generators", "plugins"}:
            return SourceKind.tooling, "package manifest or tooling directory convention"
        return SourceKind.production, "remaining source path; target membership is not inferred"

    @staticmethod
    def _syntax(facts: Facts, files: dict[str, AppleSourceFile]) -> None:
        for name, gaps, message in facts.db.execute(
            "SELECT path, parser_gaps, parser_message FROM swift_source_facts"
        ).fetchall():
            if name in files:
                files[name] = files[name].model_copy(
                    update={
                        "parsed": gaps == 0,
                        "parser_gaps": gaps,
                        "parser_message": message or None,
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
                if name in files:
                    file = files[name]
                    hit = Finding(
                        rule=rule.id,
                        span=Span(path=name, start_line=line, start_column=column),
                        message=message,
                    )
                    files[name] = file.model_copy(update={"boundaries": (*file.boundaries, hit)})
        for file in files.values():
            if file.language == "swift" and file.lint_scope and file.parser_gaps is None:
                raise MissionError(f"Swift syntax coverage is missing for {file.path}")
            files[file.path] = file.model_copy(
                update={
                    "boundaries": tuple(
                        sorted(
                            file.boundaries,
                            key=lambda hit: (
                                hit.span.start_line, hit.span.start_column, hit.rule, hit.message
                            ),
                        )
                    ),
                }
            )


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
