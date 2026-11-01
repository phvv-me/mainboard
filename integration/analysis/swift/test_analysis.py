"""Swift rules use real syntax trees, with conservative negatives and a read-only lint gate."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.analysis import Rule
from mainboard.analysis.commands.gate import Options, Pass
from mainboard.analysis.constants import LANGUAGES
from mainboard.analysis.engine.lint import Lint, load_catalog
from mainboard.analysis.engine.lint.verbs import _example_path
from mainboard.analysis.project.config import ProjectConfig
from mainboard.analysis.providers.constants import TABLES
from mainboard.analysis.providers.ducks import Ducks
from mainboard.analysis.providers.external import Table
from mainboard.analysis.providers.swift import SwiftSource
from mainboard.analysis.rules.general.apple import ObjectiveCImplementation
from mainboard.analysis.rules.swift.migration import (
    HandwrittenDatabaseDependency,
    ObjectiveCExposure,
    PlatformFrameworkImport,
    PlatformUIBoundary,
    SwiftParserCoverage,
)
from mainboard.analysis.rules.swift.presentation.ui import PrivateViewState, RedundantBodyErasure

from ...builders import git

_VIEW_SWIFT = "View.swift"

_SWIFT_RULES = (
    PrivateViewState,
    RedundantBodyErasure,
    SwiftParserCoverage,
    PlatformFrameworkImport,
    PlatformUIBoundary,
    ObjectiveCExposure,
    HandwrittenDatabaseDependency,
)


def _find(root: Path, files: Mapping[str, str], rules=_SWIFT_RULES) -> dict[str, list[tuple]]:
    for name, content in files.items():
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
    with Lint(root, rules, fixing=False) as lint:
        return lint.findings(lint.facts(root))


def test_swift_language_and_catalog_are_registered_without_changing_other_languages() -> None:
    load_catalog()
    assert LANGUAGES["SW"] == "swift"
    assert Ducks().language(".swift") == "swift"
    assert Ducks().language(".py") == "python"
    assert _example_path("swift", "struct A {}") == _VIEW_SWIFT
    assert all(Rule.defined[rule.id] is rule for rule in _SWIFT_RULES)
    assert all(rule.fixes() == "none" for rule in (*_SWIFT_RULES, ObjectiveCImplementation))
    assert all(rule.guidance and rule.references for rule in _SWIFT_RULES)


@pytest.mark.swift_native
def test_state_ownership_uses_attributes_conformance_and_full_private_access(
    tmp_path: Path,
) -> None:
    found = _find(
        tmp_path,
        {
            _VIEW_SWIFT: """import SwiftUI
struct Screen: View {
    @State var selected = false
    @State private var query = ""
    @SwiftUI.State fileprivate var count = 0
    @State private(set) var mode = 0
    @StateObject var model = Model()
    @StateObject private var correctModel = Model()
    @Binding var inherited: Bool
    // @State var comment = 0
    let text = "@State var notCode = 0"
    var body: some View { Text(text) }
}
struct Model {}
struct Ordinary { @State var notView = 0 }
"""
        },
    )
    assert not found[SwiftParserCoverage.id]
    rows = found[PrivateViewState.id]
    assert sorted(row[1] for row in rows) == [3, 5, 6, 7]
    assert all("https://developer.apple.com/" in row[3] for row in rows)


@pytest.mark.parametrize(
    "source",
    [
        "struct Screen: View { @State var selected = false }",
        "import SwiftUI\nstruct Ordinary { @State var selected = false }",
        "import SwiftUI\nstruct State {}\nstruct Screen: View { @State var selected = false }",
        "import SwiftUI\nprotocol View {}\nstruct Screen: View { @State var selected = false }",
        "import SwiftUI\nstruct Screen: View { @State private var selected = false }",
    ],
)
@pytest.mark.swift_native
def test_state_does_not_guess_unresolved_or_non_view_ownership(
    tmp_path: Path, source: str
) -> None:
    assert not _find(tmp_path, {_VIEW_SWIFT: source})[PrivateViewState.id]


@pytest.mark.swift_native
def test_erasure_review_is_limited_to_one_direct_opaque_body_expression(tmp_path: Path) -> None:
    found = _find(
        tmp_path,
        {
            _VIEW_SWIFT: """import SwiftUI
struct Direct: View { var body: some View { AnyView(Text("Hi")) } }
struct Explicit: View { var body: AnyView { AnyView(Text("Hi")) } }
struct Branches: View {
    var body: some View { if true { AnyView(Text("Hi")) } else { AnyView(EmptyView()) } }
}
struct Helper: View {
    var body: some View { Text("Hi") }
    func erase() -> AnyView { AnyView(Text("Hi")) }
}
struct Collection: View {
    let items: [AnyView] = [AnyView(Text("Hi"))]
    var body: some View { Text("Hi") }
}
struct Multiple: View { var body: some View { AnyView(Text("Hi")); EmptyView() } }
// AnyView(Text("not code"))
"""
        },
    )
    assert not found[SwiftParserCoverage.id]
    assert len(found[RedundantBodyErasure.id]) == 1
    assert "Direct.body" in found[RedundantBodyErasure.id][0][3]
    assert RedundantBodyErasure.applicability == "review"


@pytest.mark.swift_native
def test_custom_anyview_is_not_judged_as_swiftui_type_erasure(tmp_path: Path) -> None:
    found = _find(
        tmp_path,
        {
            _VIEW_SWIFT: """import SwiftUI
struct AnyView { init(_ text: Text) {} }
struct Screen: View { var body: some View { AnyView(Text("Hi")) } }
"""
        },
    )
    assert not found[RedundantBodyErasure.id]


@pytest.mark.swift_native
def test_migration_reports_real_imports_inheritance_and_objc_attributes(tmp_path: Path) -> None:
    found = _find(
        tmp_path,
        {
            "Legacy.swift": """import UIKit
import SwiftUI
#if os(macOS)
import AppKit
struct MacInput: SwiftUI.NSViewControllerRepresentable {}
#endif
@objcMembers class Old: UIViewController {
    @objc(run:) func run(_ sender: Any) {}
    let label = "@objc import UIKit UIViewRepresentable"
}
struct Input: UIViewRepresentable {}
// @objc class Fake: UIViewController {}
"""
        },
    )
    assert not found[SwiftParserCoverage.id]
    assert len(found[PlatformFrameworkImport.id]) == 2
    assert len(found[PlatformUIBoundary.id]) == 3
    assert len(found[ObjectiveCExposure.id]) == 2
    assert all(
        "https://developer.apple.com/" in row[3]
        for rule in (PlatformFrameworkImport, PlatformUIBoundary, ObjectiveCExposure)
        for row in found[rule.id]
    )


@pytest.mark.swift_native
def test_parser_gap_is_coverage_evidence_and_suppresses_partial_tree_judgments(
    tmp_path: Path,
) -> None:
    found = _find(
        tmp_path,
        {
            "Broken.swift": """import UIKit
import SwiftUI
struct Screen: View { @State var selected = false
var body: some View { Text("unfinished")
"""
        },
    )
    assert len(found[SwiftParserCoverage.id]) == 1
    assert "SwiftParser reported" in found[SwiftParserCoverage.id][0][3]
    assert all(not rows for rule, rows in found.items() if rule != SwiftParserCoverage.id)


def test_objc_inventory_uses_paths_skips_ignored_vendor_and_reports_no_ast_claim(
    tmp_path: Path,
) -> None:
    git(tmp_path, "init", "--quiet")
    found = _find(
        tmp_path,
        {
            "Legacy.m": "@implementation Legacy\n@end\n",
            "Interop.mm": "// C++ interoperability\n",
            "Header.h": "// Header alone is not an implementation\n",
            ".gitignore": "Generated/\n",
            ".gitattributes": "Vendor/** linguist-vendored\n",
            "Generated/generated.m": "",
            "Vendor/third-party.mm": "",
        },
        (ObjectiveCImplementation,),
    )
    rows = found[ObjectiveCImplementation.id]
    assert {row[0] for row in rows} == {"Legacy.m", "Interop.mm"}
    assert all("File inventory only; target membership is unknown" in row[3] for row in rows)


def test_native_rule_policy_has_explicit_path_exclusions_and_no_legacy_precedence(
    tmp_path: Path,
) -> None:
    (tmp_path / "mainboard.toml").write_text("""[analysis.rules."SW-UI0001"]
exclude = ["Generated/**"]
[analysis.rules."SW-MIG0001"]
enabled = false
""")
    (tmp_path / "pyproject.toml").write_text('[tool.mcmr.rules."PY-TEST0001"]\nenabled = false\n')
    policy = ProjectConfig.of(tmp_path)
    assert not policy.rule("SW-UI0001").accepts("Generated/View.swift")
    assert policy.rule("SW-UI0001").accepts("Sources/View.swift")
    assert not policy.rule("SW-MIG0001").enabled
    assert not policy.rule("PY-TEST0001").enabled
    (tmp_path / "pyproject.toml").write_text('[tool.mcmr.rules."SW-UI0001"]\nenabled = true\n')
    with pytest.raises(ValueError, match="configured in both"):
        ProjectConfig.of(tmp_path)


@pytest.mark.swift_native
def test_gate_is_read_only_and_reports_only_selected_swift_files(tmp_path: Path) -> None:
    for name in ("Selected.swift", "Other.swift"):
        (tmp_path / name).write_text(
            "import SwiftUI\nstruct Screen: View { @State var value = 0 }\n"
        )
    before = {file.name: file.read_bytes() for file in tmp_path.glob("*.swift")}
    with Pass(
        tmp_path, (PrivateViewState,), ["Selected.swift"], memory="1GB", fixing=True
    ) as gate:
        facts = gate.facts(tmp_path)
        assert not any(gate.repairs(facts, ("safe",)).values())
        reported = gate.reported(facts)
    ((path, line, *_),) = reported.found[PrivateViewState.id]
    assert reported.total == 1 and (path, line) == ("Selected.swift", 2)
    assert {file.name: file.read_bytes() for file in tmp_path.glob("*.swift")} == before


def test_swift_rules_do_not_run_on_python_comments_or_strings(tmp_path: Path) -> None:
    found = _find(tmp_path, {"module.py": 'text = "import UIKit @State var value: Int @objc"\n'})
    assert found == {}


def test_selected_gate_validates_rule_ids_and_skip_further_narrows() -> None:
    assert Options(rule=(PrivateViewState.id,)).selected() == [PrivateViewState]
    assert Options(skip=(PrivateViewState.id,), rule=(PrivateViewState.id,)).selected() == []
    with pytest.raises(MissionError, match="no rule .* to select"):
        Options(rule=("SW-MISSING",)).selected()
    with pytest.raises(MissionError, match="no rule .* to skip"):
        Options(skip=("SW-MISSING",)).selected()


@pytest.mark.swift_native
def test_empty_swift_is_valid_and_sql_literals_alone_are_not_database_evidence(
    tmp_path: Path,
) -> None:
    found = _find(
        tmp_path,
        {
            "Empty.swift": "",
            "Database.swift": "import SQLite3\nimport FMDB\n",
            "SwiftData.swift": (
                'import SwiftData\nimport CoreData\nlet query = "SELECT * FROM subjects"\n'
            ),
        },
        (HandwrittenDatabaseDependency, SwiftParserCoverage),
    )
    assert not found[SwiftParserCoverage.id]
    assert {row[0] for row in found[HandwrittenDatabaseDependency.id]} == {"Database.swift"}
    assert len(found[HandwrittenDatabaseDependency.id]) == 2


@pytest.mark.swift_native
def test_native_positions_use_zero_based_utf8_columns_after_non_ascii_text(tmp_path: Path) -> None:
    prefix = 'struct Screen: View { let text = "漢字"; '
    found = _find(
        tmp_path, {_VIEW_SWIFT: "import SwiftUI\n" + prefix + "@State var value = false }\n"}
    )
    row = found[PrivateViewState.id][0]
    assert row[1:3] == (2, len(prefix.encode("utf-8")))


@pytest.fixture
def normalized_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    """`SwiftSource.run` answers with the facts of one `View.swift`, so no parser executes."""
    place = ("swift:View.swift", _VIEW_SWIFT, "swift", 2, 0)
    relations: dict[str, Table] = {name: (columns, []) for name, columns in TABLES.items()}
    relations["facts"][1].append((*place, 0, ""))
    relations["states"][1].extend(
        [
            (*place, "owned", "State", 0),
            (*place, "privateOwned", "State", 1),
        ]
    )
    relations["erasures"][1].append((*place, "Screen"))
    relations["imports"][1].append((*place, "UIKit"))
    monkeypatch.setattr(SwiftSource, "run", lambda self, root: relations)


@pytest.mark.usefixtures("normalized_facts")
def test_rule_sql_runs_cross_platform_over_normalized_native_facts(tmp_path: Path) -> None:
    found = _find(tmp_path, {_VIEW_SWIFT: "// normalized-facts SQL test; no parser execution\n"})
    assert len(found[PrivateViewState.id]) == 1
    assert "`owned`" in found[PrivateViewState.id][0][3]
    assert len(found[RedundantBodyErasure.id]) == 1
    assert len(found[PlatformFrameworkImport.id]) == 1
    assert not found[SwiftParserCoverage.id]
