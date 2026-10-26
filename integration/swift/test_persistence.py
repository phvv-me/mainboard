"""SwiftData/SQLite review signals share the native parser and never rewrite persistence."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.analysis import Rule
from mainboard.analysis.engine.lint import Lint, load_catalog
from mainboard.analysis.providers.constants import TABLES
from mainboard.analysis.providers.external import Table
from mainboard.analysis.providers.swift import SwiftSource
from mainboard.analysis.rules.swift.migration import SwiftParserCoverage
from mainboard.analysis.rules.swift.persistence import (
    DirectSQLiteCall,
    OpaqueModelAttribute,
    PersistenceConstruction,
)
from mainboard.core import Runner

_RULES = (OpaqueModelAttribute, PersistenceConstruction, DirectSQLiteCall, SwiftParserCoverage)


def _find(root: Path, files: Mapping[str, str]) -> dict[str, list[tuple]]:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    with Lint(root, _RULES, fixing=False) as lint:
        return lint.findings(lint.facts(root))


def test_persistence_audits_are_registered_reviews_with_offline_guidance_and_no_fix() -> None:
    load_catalog()
    for rule in _RULES[:-1]:
        assert Rule.defined[rule.id] is rule
        assert rule.applicability == "review" and rule.fixes() == "none"
        assert rule.examples and rule.guidance and rule.references
        assert all(url.startswith("https://developer.apple.com/") for url in rule.references)
    assert "cannot decide type ownership" in OpaqueModelAttribute.guidance
    assert "cannot establish store identity" in PersistenceConstruction.guidance
    assert "neither searches arbitrary SQL strings" in DirectSQLiteCall.guidance


@pytest.mark.swift_native
def test_direct_options_on_model_properties_have_exact_utf8_positions(tmp_path: Path) -> None:
    prefix = "    @Attribute(.unique, "
    found = _find(
        tmp_path,
        {
            "Record.swift": "import SwiftData\n@Model final class Record {\n"
            + prefix
            + ".codable) var payload: Payload\n"
            + "    @Attribute(.transformable(by: Transformer.self)) var legacy: Payload\n"
            + '    @SwiftData.Attribute(.transformable(by: "Transformer")) var named: Payload\n'
            + "    @Attribute(SwiftData.Schema.Attribute.Option.codable) var external: Payload\n"
            + "    @Attribute(Schema.Attribute.Option.codable) var other: Payload\n}\n",
        },
    )
    assert not found[SwiftParserCoverage.id]
    rows = found[OpaqueModelAttribute.id]
    assert len(rows) == 5
    assert sorted(row[1] for row in rows) == [3, 4, 5, 6, 7]
    payload = next(row for row in rows if row[1] == 3)
    assert payload[1:3] == (3, len(prefix.encode("utf-8")))
    assert "Record.payload" in payload[3]
    assert sum(".transformable storage" in row[3] for row in rows) == 2


@pytest.mark.parametrize(
    "source",
    [
        "@Model class Record { @Attribute(.codable) var value: Payload }",
        "import SwiftData\nclass Record { @Attribute(.codable) var value: Payload }",
        "import SwiftData\n@Model struct Record { @Attribute(.codable) var value: Payload }",
        "import Other\n@Model class Record { @Attribute(.codable) var value: Payload }",
        "import SwiftData\n@Model class Record { @Attribute(.unique) var value: Int }",
        """import SwiftData
@Model class Record { @Attribute(originalName: ".codable") var value: Int }""",
        '''import SwiftData
let text = "@Model class Record { @Attribute(.codable) var value: Int }"''',
        "import SwiftData\n// @Model class Record { @Attribute(.codable) var value: Int }",
        "import SwiftData\n@Model class Record { @Attribute(.codable) static var value: Payload }",
        "import SwiftData\n@Model class Record { @Custom(.codable) var value: Payload }",
        "import SwiftData\n@Model class Record { @Attribute(Custom.codable) var value: Payload }",
        "import SwiftData\n@Model class Record { @Attribute(options) var value: Payload }",
        "import SwiftData\n@Model class Record { @Attribute(make(.codable)) var value: Payload }",
        """import SwiftData
@Model class Record { @Attribute(.transformable(other: 1)) var value: Payload }""",
        """import SwiftData
struct Attribute {}
@Model class Record { @Attribute(.codable) var value: Payload }""",
        """import SwiftData
struct Model {}
@Model class Record { @Attribute(.codable) var value: Payload }""",
        """import SwiftData
enum SwiftData {}
@SwiftData.Model class Record { @SwiftData.Attribute(.codable) var value: Payload }""",
        """import SwiftData
enum Schema {}
@Model class Record { @Attribute(Schema.Attribute.Option.codable) var value: Payload }""",
    ],
)
@pytest.mark.swift_native
def test_options_exclude_nonmodels_other_apis_comments_strings_and_shadows(
    tmp_path: Path, source: str
) -> None:
    found = _find(tmp_path, {"Custom.swift": source})
    assert not found[SwiftParserCoverage.id]
    assert not found[OpaqueModelAttribute.id]


@pytest.mark.swift_native
def test_qualified_model_options_survive_leaf_shadows_but_not_module_shadows(
    tmp_path: Path,
) -> None:
    found = _find(
        tmp_path,
        {
            "Record.swift": """import SwiftData
@SwiftData.Model final class Record {
    @SwiftData.Attribute(SwiftData.Schema.Attribute.Option.codable) var value: Payload
}
""",
            "Custom.swift": "struct Model {}\nstruct Attribute {}\nstruct Schema {}\n",
        },
    )
    assert not found[SwiftParserCoverage.id]
    assert len(found[OpaqueModelAttribute.id]) == 1
    (tmp_path / "Namespace.swift").write_text("enum SwiftData {}")
    assert not _find(tmp_path, {})[OpaqueModelAttribute.id]


@pytest.mark.swift_native
def test_current_swift_syntax_and_unicode_are_preserved(tmp_path: Path) -> None:
    prefix = '    let text = "漢字"; '
    found = _find(
        tmp_path,
        {
            "Current.swift": """import SwiftData
@Model final class Record {
    @Attribute(.codable) var value: Payload
    isolated deinit {}
}
nonisolated func create() throws(Failure) {
"""
            + prefix
            + "_ = ModelContext(container)\n}\n",
        },
    )
    assert not found[SwiftParserCoverage.id]
    assert len(found[OpaqueModelAttribute.id]) == 1
    row = found[PersistenceConstruction.id][0]
    assert row[1:3] == (7, len((prefix + "_ = ").encode("utf-8")))


@pytest.mark.swift_native
def test_named_constructors_and_explicit_initializers_are_each_reported_once(
    tmp_path: Path,
) -> None:
    found = _find(
        tmp_path,
        {
            "Owner.swift": """import SwiftData
let first = try ModelContainer(for: Record.self)
let second = ModelContext(first)
let third = try SwiftData.ModelContainer(for: Record.self)
let fourth = SwiftData.ModelContext(third)
let fifth = try ModelContainer.init(for: Record.self)
let sixth = SwiftData.ModelContext.init(fifth)
""",
        },
    )
    assert not found[SwiftParserCoverage.id]
    rows = found[PersistenceConstruction.id]
    assert len(rows) == 6
    assert sorted(row[1] for row in rows) == list(range(2, 8))
    assert sum("`ModelContainer` construction" in row[3] for row in rows) == 3
    assert sum("`ModelContext` construction" in row[3] for row in rows) == 3
    assert all("does not prove a duplicate writer" in row[3] for row in rows)


@pytest.mark.parametrize(
    "source",
    [
        "let context = ModelContext(container)",
        'import SwiftData\nlet text = "ModelContainer(for: Record.self)"',
        "import SwiftData\n// let context = ModelContext(container)",
        "import SwiftData\nlet context = Custom.ModelContext(container)",
        "import SwiftData\nlet context: ModelContext = .init(container)",
        "import SwiftData\nlet context = Alias(container)",
        "import SwiftData\nlet context = container.mainContext",
        "import SwiftData\nstruct ModelContext {}\nlet context = ModelContext(container)",
        "import SwiftData\nfunc ModelContext() {}\nlet context = ModelContext()",
        "import SwiftData\nlet ModelContext = factory\nlet context = ModelContext()",
        "import SwiftData\nfunc create(ModelContext: () -> Int) { let context = ModelContext() }",
        """import SwiftData
struct Owner<ModelContext> { func make() { let context = ModelContext() } }""",
        "import SwiftData\nenum SwiftData {}\nlet context = SwiftData.ModelContext(container)",
    ],
)
@pytest.mark.swift_native
def test_construction_excludes_strings_factories_other_receivers_and_known_shadows(
    tmp_path: Path, source: str
) -> None:
    found = _find(tmp_path, {"Owner.swift": source})
    assert not found[SwiftParserCoverage.id]
    assert not found[PersistenceConstruction.id]


@pytest.mark.swift_native
def test_cross_file_shadow_excludes_bare_but_not_module_qualified_construction(
    tmp_path: Path,
) -> None:
    found = _find(
        tmp_path,
        {
            "Owner.swift": """import SwiftData
let a = ModelContext(container)
let b = SwiftData.ModelContext.init(container)
""",
            "Custom.swift": "typealias ModelContext = Custom\nstruct Custom {}",
        },
    )
    assert not found[SwiftParserCoverage.id]
    assert [row[1] for row in found[PersistenceConstruction.id]] == [3]


@pytest.mark.swift_native
def test_sqlite_signal_is_finite_call_syntax_with_an_import_boundary(tmp_path: Path) -> None:
    found = _find(
        tmp_path,
        {
            "Import.swift": """import SQLite3
let first = sqlite3_exec(db, text, nil, nil, nil)
let second = SQLite3.sqlite3_prepare_v2(db, text, -1, nil, nil)
let third = sqlite3_step(statement)
let fourth = sqlite3_prepare(db, text, -1, nil, nil)
let fifth = sqlite3_prepare_v3(db, text, -1, 0, nil, nil)
let notExecution = sqlite3_column_text(statement, 0)
let reference = sqlite3_step
""",
        },
    )
    assert not found[SwiftParserCoverage.id]
    rows = found[DirectSQLiteCall.id]
    assert len(rows) == 5 and sorted(row[1] for row in rows) == list(range(2, 7))
    assert any(row[1:3] == (2, 12) for row in rows)
    assert all("read/write mode" in row[3] for row in rows)


@pytest.mark.parametrize(
    "source",
    [
        'let value = sqlite3_exec(db, "SELECT 1", nil, nil, nil)',
        'import SQLite3\nlet query = "SELECT * FROM Record"',
        'import SQLite3\nlet query = #"sqlite3_exec(db, "SELECT 1")"#',
        "import SQLite3\n// sqlite3_step(statement)",
        "import SQLite3\nlet value = Other.sqlite3_step(statement)",
        "import SQLite3\nfunc sqlite3_step() {}\nlet value = sqlite3_step()",
        "import SQLite3\nlet sqlite3_step = custom\nlet value = sqlite3_step()",
        "import SQLite3\nfunc run(sqlite3_step: () -> Int) { let value = sqlite3_step() }",
        "import SQLite3\nenum SQLite3 {}\nlet value = SQLite3.sqlite3_step(statement)",
    ],
)
@pytest.mark.swift_native
def test_sqlite_excludes_arbitrary_text_unrelated_calls_and_project_shadows(
    tmp_path: Path, source: str
) -> None:
    found = _find(tmp_path, {"Import.swift": source})
    assert not found[SwiftParserCoverage.id]
    assert not found[DirectSQLiteCall.id]


@pytest.mark.swift_native
def test_incomplete_source_has_only_coverage_evidence(tmp_path: Path) -> None:
    found = _find(
        tmp_path,
        {
            "Broken.swift": """import SwiftData
@Model class Record { @Attribute(.codable) var value: Payload"""
        },
    )
    assert len(found[SwiftParserCoverage.id]) == 1
    assert all(not found[rule.id] for rule in _RULES[:-1])


@pytest.mark.swift_native
def test_reviews_never_rewrite_schema_queries_or_legitimate_test_containers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "Fixture.swift"
    path.write_text(
        """import SwiftData
@Model class Record { @Attribute(.codable) var value: Payload }
let fixture = try ModelContainer(for: Record.self)
import SQLite3
let status = sqlite3_exec(db, "SELECT 1", nil, nil, nil)
"""
    )
    before = path.read_bytes()
    with Lint(tmp_path, _RULES, fixing=True) as lint:
        facts = lint.facts(tmp_path)
        found = lint.findings(facts)
        assert all(len(found[rule.id]) == 1 for rule in _RULES[:-1])
        assert not any(lint.repairs(facts, ("safe", "review")).values())
    assert path.read_bytes() == before


def test_a_selected_persistence_review_refuses_a_missing_native_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(runner: Runner) -> Path:
        raise MissionError("Swift analysis requires the selected Xcode Swift toolchain")

    monkeypatch.setattr(SwiftSource, "_helper", staticmethod(missing))
    with pytest.raises(MissionError, match="Swift analysis requires"):
        _find(tmp_path, {"Store.swift": "import SwiftData\nlet context = ModelContext(container)"})


def test_persistence_query_contracts_are_portable_over_normalized_syntax_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    place = ("swift:Store.swift", "Store.swift", "swift", 4, 10)
    relations: dict[str, Table] = {name: (columns, []) for name, columns in TABLES.items()}
    relations["facts"][1].append((*place, 0, ""))
    relations["imports"][1].extend([(*place, "SwiftData"), (*place, "SQLite3")])
    relations["attributes"][1].append((*place, 12, "Record", "Model", -1, "", "", 0, 0))
    relations["attribute_options"][1].extend(
        [
            (*place, 30, "value", 12, "Record", "class", 0, "Attribute", 0, "codable", "", 0),
            (*place, 40, "static", 12, "Record", "class", 1, "Attribute", 0, "codable", "", 0),
        ]
    )
    relations["calls"][1].extend(
        [
            (*place, "ModelContext", "", 0),
            (*place, "init", "SwiftData.ModelContainer", 0),
            (*place, "ModelContext", "", 1),
            (*place, "sqlite3_exec", "SQLite3", 0),
            (*place, "sqlite3_step", "Other", 0),
        ]
    )
    monkeypatch.setattr(SwiftSource, "run", lambda self, root: relations)
    found = _find(tmp_path, {"Store.swift": "// portable normalized-fact rule query test"})
    assert len(found[OpaqueModelAttribute.id]) == 1
    assert len(found[PersistenceConstruction.id]) == 2
    assert len(found[DirectSQLiteCall.id]) == 1
    assert not found[SwiftParserCoverage.id]


def test_persistence_reviews_do_not_inspect_non_swift_source(tmp_path: Path) -> None:
    assert not _find(
        tmp_path, {"module.py": 'text = "ModelContext @Attribute(.codable) sqlite3_exec()"'}
    )
