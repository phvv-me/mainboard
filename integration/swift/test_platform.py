"""Finite UIKit references are placed syntax signals, never UI/NS prefix guesses."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from mainboard.analysis import Rule
from mainboard.analysis.engine.lint import Lint, load_catalog
from mainboard.analysis.rules.swift.migration import SwiftParserCoverage
from mainboard.analysis.rules.swift.platform import UIKitSymbolReference

_RULES = (UIKitSymbolReference, SwiftParserCoverage)


def _find(root: Path, files: Mapping[str, str]) -> dict[str, list[tuple]]:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    with Lint(root, _RULES, fixing=False) as lint:
        return lint.findings(lint.facts(root))


def test_finite_uikit_inventory_is_a_review_with_bridge_guidance_and_no_fix() -> None:
    load_catalog()
    assert Rule.defined[UIKitSymbolReference.id] is UIKitSymbolReference
    assert UIKitSymbolReference.applicability == "review"
    assert UIKitSymbolReference.fixes() == "none"
    assert UIKitSymbolReference.examples and UIKitSymbolReference.references
    assert "IME" in UIKitSymbolReference.guidance
    assert "re-export" in UIKitSymbolReference.guidance
    assert "not resolve" in UIKitSymbolReference.guidance
    assert all(
        reference.startswith("https://developer.apple.com/")
        for reference in UIKitSymbolReference.references
    )


@pytest.mark.swift_native
def test_transitive_uikit_calls_annotations_and_member_bases_have_source_positions(
    tmp_path: Path,
) -> None:
    found = _find(
        tmp_path,
        {
            "Radical.swift": """import Foundation
import SVGKit
let scale = UIScreen.main.scale
let image = UIImage(contentsOfFile: "radical.png")
func render(_ image: UIImage?) -> UIImage? { image }
"""
        },
    )
    assert not found[SwiftParserCoverage.id]
    rows = found[UIKitSymbolReference.id]
    assert len(rows) == 4
    assert sorted(row[1] for row in rows) == [3, 4, 5, 5]
    assert any(row[1:3] == (4, 12) for row in rows)
    assert any("container geometry" in row[3] for row in rows)
    assert any("retain required image processing" in row[3] for row in rows)
    assert all("not resolved types" in row[3] for row in rows)


@pytest.mark.swift_native
def test_qualified_references_are_reported_once_per_known_type_occurrence(tmp_path: Path) -> None:
    found = _find(
        tmp_path,
        {
            "Input.swift": """import UIKit
let button = UIKit.UIButton()
var field: UIKit.UITextField?
let screen = UIKit.UIScreen.main
let metatype = UIKit.UIImage.self
"""
        },
    )
    assert not found[SwiftParserCoverage.id]
    rows = found[UIKitSymbolReference.id]
    assert len(rows) == 4
    assert sorted(row[1] for row in rows) == [2, 3, 4, 5]


@pytest.mark.swift_native
def test_each_declared_finite_symbol_is_a_candidate_without_an_explicit_import(
    tmp_path: Path,
) -> None:
    symbols = (
        "UIView",
        "UIViewController",
        "UIButton",
        "UILabel",
        "UITextField",
        "UIImage",
        "UIImageView",
        "UIColor",
        "UIFont",
        "UIScreen",
        "UIApplication",
        "UIWindow",
        "UINavigationController",
        "UISplitViewController",
        "UIStackView",
        "UITableView",
        "UITableViewController",
        "UIScrollView",
    )
    source = "\n".join(f"var value{index}: {name}?" for index, name in enumerate(symbols))
    found = _find(tmp_path, {"Inventory.swift": source})
    assert not found[SwiftParserCoverage.id]
    rows = found[UIKitSymbolReference.id]
    assert len(rows) == len(symbols)
    assert {row[1] for row in rows} == set(range(1, len(symbols) + 1))
    assert all("review" in row[3] and "bridges" in row[3] for row in rows)


@pytest.mark.parametrize(
    "source",
    [
        '// UIImage() UIScreen.main\nlet text = "UIButton UILabel UITextField"',
        'let text = #"UIImage(contentsOfFile: "/radical.png")"#',
        'import SwiftUI\nlet image = Image("radical")\nlet color = Color.blue',
        "let view = UICustomView()\nlet controller = UIFlowController()\nlet image = NSImage()",
        "let image = Other.UIImage()\nvar field: Custom.UITextField?",
        "struct UIImage {}\nlet image = UIImage()",
        'let UIImage = { "custom" }\nlet image = UIImage()',
        'func UIImage() -> String { "custom" }\nlet image = UIImage()',
        "func create(UIImage: () -> String) -> String { UIImage() }",
        "struct Custom<UIImage> { var image: UIImage }",
        "typealias UIImage = Custom\nstruct Custom {}\nvar image: UIImage?",
        "enum UIKit { static func UIImage() {} }\nlet image = UIKit.UIImage()",
        "import UIKit\nenum UIKit { class UIButton {} }\nvar button: UIKit.UIButton?",
    ],
)
@pytest.mark.swift_native
def test_inventory_excludes_comments_strings_unrelated_names_receivers_and_shadows(
    tmp_path: Path, source: str
) -> None:
    found = _find(tmp_path, {"Custom.swift": source})
    assert not found[SwiftParserCoverage.id]
    assert not found[UIKitSymbolReference.id]


@pytest.mark.swift_native
def test_project_shadow_suppresses_bare_but_not_genuine_qualified_reference(
    tmp_path: Path,
) -> None:
    found = _find(
        tmp_path,
        {
            "Input.swift": "let first = UIImage()\nlet second = UIKit.UIImage()",
            "Custom.swift": "struct UIImage {}",
        },
    )
    assert not found[SwiftParserCoverage.id]
    rows = found[UIKitSymbolReference.id]
    assert len(rows) == 1
    assert rows[0][0:2] == ("Input.swift", 2)


@pytest.mark.swift_native
def test_migration_audit_keeps_a_required_input_bridge_and_media_code_unchanged(
    tmp_path: Path,
) -> None:
    path = tmp_path / "Bridge.swift"
    path.write_text(
        """import UIKit
class JapaneseInput: UITextField {}
let image = UIImage(contentsOfFile: "cached-radical.png")
"""
    )
    before = path.read_bytes()
    with Lint(tmp_path, _RULES, fixing=True) as lint:
        facts = lint.facts(tmp_path)
        assert len(lint.findings(facts)[UIKitSymbolReference.id]) == 2
        assert not any(lint.repairs(facts, ("safe", "review")).values())
    assert path.read_bytes() == before


@pytest.mark.swift_native
def test_incomplete_syntax_is_coverage_only(tmp_path: Path) -> None:
    found = _find(tmp_path, {"Broken.swift": "let image = UIImage(\n"})
    assert len(found[SwiftParserCoverage.id]) == 1
    assert not found[UIKitSymbolReference.id]
