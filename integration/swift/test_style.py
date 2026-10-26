"""Migration audits read native syntax, preserve source and reject name-only false positives."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from mainboard.analysis import Rule
from mainboard.analysis.commands.gate import Pass
from mainboard.analysis.engine.lint import Lint, load_catalog
from mainboard.analysis.rules.swift.migration import SwiftParserCoverage
from mainboard.analysis.rules.swift.presentation.style import (
    LegacyNavigationView,
    ObservableModelMigration,
)

_RULES = (LegacyNavigationView, ObservableModelMigration, SwiftParserCoverage)


def _find(root: Path, files: Mapping[str, str]) -> dict[str, list[tuple]]:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    with Lint(root, _RULES, fixing=False) as lint:
        return lint.findings(lint.facts(root))


def test_swift_migration_audits_are_registered_review_only_with_offline_guidance() -> None:
    load_catalog()
    for rule in (LegacyNavigationView, ObservableModelMigration):
        assert Rule.defined[rule.id] is rule
        assert rule.applicability == "review"
        assert rule.fixes() == "none"
        assert rule.guidance and rule.references and rule.examples
        assert all(
            reference.startswith("https://developer.apple.com/") for reference in rule.references
        )
    assert "external" in ObservableModelMigration.guidance
    assert "incremental" in ObservableModelMigration.guidance
    assert "restoration" in LegacyNavigationView.guidance


@pytest.mark.swift_native
def test_navigation_construction_requires_syntax_and_reports_each_actual_call(
    tmp_path: Path,
) -> None:
    found = _find(
        tmp_path,
        {
            "View.swift": """import SwiftUI
struct Screen: View {
    var body: some View {
        NavigationView { Text("Review") }
    }
    func sidebar() -> some View {
        SwiftUI.NavigationView { Text("Subjects") }
    }
    // NavigationView { Text("comment") }
    let note = "NavigationView()"
}
"""
        },
    )
    assert not found[SwiftParserCoverage.id]
    assert [row[1] for row in found[LegacyNavigationView.id]] == [4, 7]
    assert not found[ObservableModelMigration.id]


@pytest.mark.parametrize(
    "source",
    [
        'let content = NavigationView { Text("No import") }',
        'import SwiftUI\nlet content = NavigationStack { Text("Review") }',
        (
            """import SwiftUI
let content = NavigationSplitView { Text("List") } detail: { Text("Review") }"""
        ),
        'import SwiftUI\nlet content = Custom.NavigationView { Text("Custom") }',
        "import SwiftUI\nstruct NavigationView {}\nlet content = NavigationView()",
        (
            """import SwiftUI
func NavigationView() -> Text { Text("Custom") }
let content = NavigationView()"""
        ),
        'import SwiftUI\nlet NavigationView = { Text("Custom") }\nlet content = NavigationView()',
        "import SwiftUI\nfunc content(NavigationView: () -> Text) -> Text { NavigationView() }",
        "import SwiftUI\nstruct Screen<NavigationView> { func content() { NavigationView() } }",
        (
            """import SwiftUI
struct Screen { let NavigationView: () -> Text;
func content() -> Text { NavigationView() } }"""
        ),
        (
            """import SwiftUI
enum SwiftUI { static func NavigationView() {} }
let content = SwiftUI.NavigationView()"""
        ),
        'import SwiftUI\n// NavigationView()\nlet text = "NavigationView()"',
    ],
)
@pytest.mark.swift_native
def test_navigation_ignores_modern_controls_unimported_and_shadowed_names(
    tmp_path: Path, source: str
) -> None:
    found = _find(tmp_path, {"View.swift": source})
    assert not found[SwiftParserCoverage.id]
    assert not found[LegacyNavigationView.id]


@pytest.mark.swift_native
def test_qualified_navigation_survives_an_unqualified_type_shadow(tmp_path: Path) -> None:
    found = _find(
        tmp_path,
        {
            "View.swift": (
                'import SwiftUI\nlet content = SwiftUI.NavigationView { Text("Review") }'
            ),
            "Custom.swift": "struct NavigationView {}",
        },
    )
    assert not found[SwiftParserCoverage.id]
    assert len(found[LegacyNavigationView.id]) == 1


@pytest.mark.swift_native
def test_observation_audit_requires_instance_published_state_and_emits_one_row_per_class(
    tmp_path: Path,
) -> None:
    found = _find(
        tmp_path,
        {
            "Model.swift": """import SwiftUI
import Combine
final class ReviewModel: ObservableObject {
    @Published var answer = ""
    @Published var reviewing = false
}
final class QualifiedModel: Combine.ObservableObject {
    @Combine.Published var completed = 0
}
final class SwiftUIModel: SwiftUI.ObservableObject {
    @SwiftUI.Published var completed = 0
}
"""
        },
    )
    assert not found[SwiftParserCoverage.id]
    rows = found[ObservableModelMigration.id]
    assert len(rows) == 3
    assert {row[1] for row in rows} == {3, 7, 10}
    assert all("external protocol" in row[3] for row in rows)
    assert not found[LegacyNavigationView.id]


@pytest.mark.parametrize(
    "source",
    [
        "import Combine\nfinal class StreamModel: ObservableObject { @Published var value = 0 }",
        "import SwiftUI\nfinal class Model: ObservableObject { var value = 0 }",
        "import SwiftUI\nfinal class Model: ObservableObject { @Published static var value = 0 }",
        "import SwiftUI\nfinal class Model { @Published var value = 0 }",
        "import SwiftUI\nstruct Model: ObservableObject { @Published var value = 0 }",
        "import SwiftUI\nfinal class Model: Other.ObservableObject { @Published var value = 0 }",
        "import SwiftUI\nfinal class Model: ObservableObject { @Other.Published var value = 0 }",
        (
            """import SwiftUI
@Observable final class Model: ObservableObject { @Published var value = 0 }"""
        ),
        "import SwiftUI\n@Observable final class Model { var value = 0 }",
        (
            '''import SwiftUI
// class Model: ObservableObject { @Published var value = 0 }
let text = "ObservableObject @Published"'''
        ),
        (
            """import SwiftUI
protocol ObservableObject {}
final class Model: ObservableObject { @Published var value = 0 }"""
        ),
        (
            """import SwiftUI
@propertyWrapper struct Published<Value> { var wrappedValue: Value }
final class Model: ObservableObject { @Published var value = 0 }"""
        ),
        (
            """import SwiftUI
enum Combine { protocol ObservableObject {} }
final class Model: Combine.ObservableObject { @Published var value = 0 }"""
        ),
    ],
)
@pytest.mark.swift_native
def test_observation_audit_does_not_assume_every_model_should_migrate(
    tmp_path: Path, source: str
) -> None:
    found = _find(tmp_path, {"Model.swift": source})
    assert not found[SwiftParserCoverage.id]
    assert not found[ObservableModelMigration.id]


@pytest.mark.swift_native
def test_project_wide_shadows_suppress_unqualified_observation_names(tmp_path: Path) -> None:
    found = _find(
        tmp_path,
        {
            "Model.swift": (
                "import SwiftUI\nfinal class Model: ObservableObject { @Published var value = 0 }"
            ),
            "Custom.swift": (
                "typealias ObservableObject = CustomProtocol\nprotocol CustomProtocol {}"
            ),
        },
    )
    assert not found[SwiftParserCoverage.id]
    assert not found[ObservableModelMigration.id]


@pytest.mark.swift_native
def test_audit_findings_offer_no_repairs_and_leave_the_selected_source_unchanged(
    tmp_path: Path,
) -> None:
    source = tmp_path / "View.swift"
    source.write_text('import SwiftUI\nlet content = NavigationView { Text("Reviews") }\n')
    before = source.read_bytes()
    with Pass(tmp_path, (LegacyNavigationView,), [source.name], memory="1GB", fixing=True) as gate:
        facts = gate.facts(tmp_path)
        assert not any(gate.repairs(facts, ("safe", "review")).values())
        assert gate.reported(facts).total == 1
    assert source.read_bytes() == before


@pytest.mark.swift_native
def test_incomplete_native_syntax_emits_coverage_only(tmp_path: Path) -> None:
    found = _find(
        tmp_path,
        {"Broken.swift": "import SwiftUI\nlet content = NavigationView {\n"},
    )
    assert len(found[SwiftParserCoverage.id]) == 1
    assert not found[LegacyNavigationView.id]
    assert not found[ObservableModelMigration.id]
