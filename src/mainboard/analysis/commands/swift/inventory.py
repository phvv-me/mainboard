"""Account for Apple source size and placed migration evidence without deciding deletability."""

import hashlib
from collections.abc import Callable, Set
from contextlib import closing
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Annotated

from cyclopts import Parameter

from ....core.errors import MissionError
from ....log import logger
from ...engine.facts import Budget, Facts, lint_excludes, marked, project_files, vendored
from ...findings import Finding, Span
from ...providers.swift import SwiftSource, SwiftToolchain
from ...rules.general.apple import ObjectiveCImplementation
from ...rules.swift.migration import (
    HandwrittenDatabaseDependency,
    ObjectiveCExposure,
    PlatformFrameworkImport,
    PlatformUIBoundary,
    SwiftParserCoverage,
)
from ...rules.swift.persistence import (
    DirectSQLiteCall,
    OpaqueModelAttribute,
    PersistenceConstruction,
)
from ...rules.swift.platform import UIKitSymbolReference
from ...rules.swift.presentation.style import LegacyNavigationView, ObservableModelMigration
from ...rules.swift.presentation.ui import PrivateViewState, RedundantBodyErasure
from ..catalog import RuleEntry
from ..graphs.scope import Scope
from .constants import LANGUAGES
from .models import AppleSourceFile, Condition, SourceKind, SwiftInventory, SyntaxStatus

_SOURCE_CONDITIONS_SQL = (
    "SELECT path, start_line, start_column, condition FROM swift_source_conditions "
    "ORDER BY path, start_line, start_column"
)
_PARSER_PROGRESS_SQL = "SELECT path, parser_gaps, parser_message FROM swift_source_facts"
_SOURCE_IMPORTS_SQL = "SELECT DISTINCT path, name FROM swift_source_imports ORDER BY path, name"

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


# What a path under one of these folders, or a file under these names, is for.
_TEST_FOLDERS = frozenset(
    {"tests", "test", "uitests", "apptests", "appstorescreenshots", "testsupport"}
)
_TOOLING_FOLDERS = frozenset({"tools", "scripts", "generators", "plugins"})


def _appended(
    files: dict[str, AppleSourceFile], name: str, *, field: str, value: str | Condition | Finding
) -> None:
    """Add `value` to the tuple `field` of the file `name`, a file outside `files` ignored."""
    if name in files:
        file = files[name]
        files[name] = file.model_copy(update={field: (*getattr(file, field), value)})


class SwiftSourceInventory:
    """The source listing used by analysis, retaining excluded code for size accounting."""

    @cached_property
    def excluded(self) -> Callable[[str], bool]:
        """Whether the analysis lint scope leaves a path out."""
        return lint_excludes(self.scope.root)

    @cached_property
    def names(self) -> list[str]:
        """Every Apple source file of the project, in or out of the scope."""
        return [n for n in project_files(self.scope.root) if Path(n).suffix.lower() in LANGUAGES]

    @cached_property
    def vendor(self) -> Set[str]:
        """The project's files git marks vendored."""
        return vendored(self.scope.root, self.names)

    @staticmethod
    def _boundaries(facts: Facts, files: dict[str, AppleSourceFile]) -> list[Finding]:
        """Place each boundary rule's findings on their files, and answer the rest.

        The rest are parser coverage findings about files beside the reported directory.
        """
        context: list[Finding] = []
        for rule in (specification() for specification in _RULES):
            statement = facts.db.execute(rule.statement(), rule.rows().parameters)
            for name, line, column, message in statement.fetchall():
                span = Span(path=name, start_line=line, start_column=column)
                hit = Finding(rule=rule.id, span=span, message=message)
                if name in files:
                    _appended(files, name, field="boundaries", value=hit)
                elif rule.id == SwiftParserCoverage.id:
                    context.append(hit)
        return context

    @staticmethod
    def _conditions(facts: Facts, files: dict[str, AppleSourceFile]) -> None:
        """Note every `#if`/`#elseif` each file holds, in source order."""
        for name, line, column, expression in facts.db.execute(_SOURCE_CONDITIONS_SQL).fetchall():
            condition = Condition(
                path=name, start_line=line, start_column=column, expression=expression
            )
            _appended(files, name, field="conditions", value=condition)

    @staticmethod
    def _coverage(facts: Facts, files: dict[str, AppleSourceFile]) -> None:
        """Note how far the parser read each file."""
        for name, gaps, message in facts.db.execute(_PARSER_PROGRESS_SQL).fetchall():
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

    @staticmethod
    def _imports(facts: Facts, files: dict[str, AppleSourceFile]) -> None:
        """Note the modules each file imports, each once, by name."""
        for name, module in facts.db.execute(_SOURCE_IMPORTS_SQL).fetchall():
            _appended(files, name, field="imports", value=module)

    @staticmethod
    def _kind(name: str, vendor: Set[str], *, generated: Set[str]) -> tuple[SourceKind, str]:
        """What `name` is for, and the convention that says so; the first that holds wins."""
        path = PurePosixPath(name)
        parts = {part.casefold() for part in path.parts[:-1]}
        leaf = path.name.casefold()
        conventions = [
            (name in vendor, SourceKind.vendored, "git linguist-vendored attribute"),
            (name in generated, SourceKind.generated, "git linguist-generated attribute"),
            (
                "generated" in parts or leaf.endswith((".pb.swift", "+generated.swift")),
                SourceKind.generated,
                "Generated path or generated filename convention",
            ),
            (
                bool(parts & _TEST_FOLDERS) or leaf.endswith("tests.swift"),
                SourceKind.test,
                "test directory or Tests.swift filename convention",
            ),
            (
                leaf == "package.swift" or bool(parts & _TOOLING_FOLDERS),
                SourceKind.tooling,
                "package manifest or tooling directory convention",
            ),
        ]
        production = (
            SourceKind.production,
            "remaining source path; target membership is not inferred",
        )
        return next(((kind, why) for holds, kind, why in conventions if holds), production)

    @staticmethod
    def _ordered(files: dict[str, AppleSourceFile]) -> None:
        """Refuse a linted Swift file the parser never read; order each file's boundaries."""
        for file in list(files.values()):
            if file.language == "swift" and file.lint_scope and file.parser_gaps is None:
                raise MissionError(f"Swift syntax coverage is missing for {file.path}")
            ordered = sorted(
                file.boundaries,
                key=lambda hit: (
                    hit.span.start_line,
                    hit.span.start_column,
                    hit.rule,
                    hit.message,
                ),
            )
            files[file.path] = file.model_copy(update={"boundaries": tuple(ordered)})

    @staticmethod
    def _recorded(
        path: Path, name: str, *, kind: SourceKind, reason: str, lint_scope: bool
    ) -> AppleSourceFile:
        """The file at `path` measured, its syntax pending a parse when it is linted Swift."""
        data = path.read_bytes()
        if path.suffix.lower() != ".swift":
            status, why = (
                SyntaxStatus.not_swift,
                "Objective-C file inventory only; no Objective-C parser",
            )
        elif lint_scope:
            status, why = SyntaxStatus.pending, None
        else:
            status, why = (
                SyntaxStatus.excluded,
                "Excluded by vendor attribute or analysis lint scope",
            )
        return AppleSourceFile(
            path=name,
            language=LANGUAGES[path.suffix.lower()],
            kind=kind,
            classification=reason,
            lint_scope=lint_scope,
            byte_count=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            physical_lines=len(data.splitlines()),
            nonblank_lines=sum(bool(line.strip()) for line in data.splitlines()),
            syntax_status=status,
            syntax_reason=why,
        )

    @staticmethod
    def _syntax(
        facts: Facts, files: dict[str, AppleSourceFile]
    ) -> tuple[SwiftToolchain, tuple[Finding, ...]]:
        """Fill `files` with what the parser read; answer its toolchain and the other findings.

        The other findings are those about files beside the reported directory.
        """
        SwiftSourceInventory._coverage(facts, files)
        SwiftSourceInventory._imports(facts, files)
        SwiftSourceInventory._conditions(facts, files)
        context = SwiftSourceInventory._boundaries(facts, files)
        SwiftSourceInventory._ordered(files)
        context.sort(key=lambda hit: hit.span.path)
        return SwiftSourceInventory._toolchain(facts), tuple(context)

    @staticmethod
    def _toolchain(facts: Facts) -> SwiftToolchain:
        """The compiler and SDK the parser ran under, refusing missing or ambiguous provenance."""
        metadata = facts.db.execute("SELECT * FROM swift_source_toolchain")
        columns = [column[0] for column in metadata.description]
        rows = metadata.fetchall()
        if len(rows) != 1:
            raise MissionError("Swift syntax toolchain provenance is missing or ambiguous")
        return SwiftToolchain.model_validate(dict(zip(columns, rows[0], strict=True)))

    def __init__(self, scope: Scope) -> None:
        self.scope = scope

    def _listed(self) -> dict[str, AppleSourceFile]:
        """Every Apple source file in scope, measured and classified, a link refused."""
        root = self.scope.root
        generated = marked(root, self.names, "linguist-generated")
        files: dict[str, AppleSourceFile] = {}
        for name in sorted(self.names):
            if self.scope.prefix and not name.startswith(self.scope.prefix + "/"):
                continue
            if (root / name).is_symlink():
                raise MissionError(f"source inventory refuses a symbolic link: {root / name}")
            kind, reason = SwiftSourceInventory._kind(name, self.vendor, generated=generated)
            lint_scope = name not in self.vendor and not self.excluded(name)
            files[name] = SwiftSourceInventory._recorded(
                root / name, name, kind=kind, reason=reason, lint_scope=lint_scope
            )
        return files

    def _parsed(
        self, files: dict[str, AppleSourceFile], *, memory: str
    ) -> tuple[SwiftToolchain | None, tuple[Finding, ...]]:
        """Parse the project's linted Swift when the scope holds any; nothing to parse otherwise.

        The parse reads the whole project for declaration context, so a link anywhere in it is
        refused.
        """
        if not any(file.language == "swift" and file.lint_scope for file in files.values()):
            return None, ()
        root = self.scope.root
        for name in self.names:
            swift = Path(name).suffix.lower() == ".swift"
            linted = swift and name not in self.vendor and not self.excluded(name)
            if linted and (root / name).is_symlink():
                raise MissionError(f"Swift inventory refuses a symbolic link: {name}")
        with closing(Budget(memory=memory)) as budget, closing(Facts(root, budget)) as facts:
            provider = SwiftSource()
            provider.register(facts, provider.run(root))
            return SwiftSourceInventory._syntax(facts, files)

    def read(self, *, memory: str = "2GB") -> SwiftInventory:
        """The inventory of the scope, every file read twice so one that changed is refused."""
        root = self.scope.root
        files = self._listed()
        toolchain, context_issues = self._parsed(files, memory=memory)
        legacy = ObjectiveCImplementation()
        hit = legacy.message or legacy.summary()
        for name, file in files.items():
            if file.language != "swift" and file.lint_scope:
                finding = Finding(rule=legacy.id, span=Span(path=name, start_line=1), message=hit)
                files[name] = file.model_copy(update={"boundaries": (finding,)})
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
    logger.result(report.render(as_json=as_json))
