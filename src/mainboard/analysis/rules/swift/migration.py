"""Platform and Objective-C interoperability boundaries to review, never blanket API bans."""

from ...catalog import Example, Rule
from .provenance import apple_review


class SwiftParserCoverage(Rule, id="SW-PARSE0001"):
    """A Swift file the selected Swift toolchain cannot parse completely."""

    query = "SELECT * FROM swift_source_facts WHERE parser_gaps > 0"
    provenance = apple_review("SwiftParser diagnostics and exact requested-file coverage protocol")
    message = (
        "SwiftParser reported {parser_gaps} syntax diagnostics; SwiftUI checks skipped "
        "this file. Verify with the selected Swift compiler. {parser_message}"
    )
    guidance = (
        "Build the relevant target with its selected Swift compiler. SwiftParser comes from "
        "that same selected toolchain. Its diagnostics establish a parsing problem, not a "
        "type-checking or deployment-target result. No style or migration judgment is made "
        "from a partial syntax tree. Check the reported syntax and selected toolchain first."
    )
    references = ("https://www.swift.org/documentation/",)


class PlatformFrameworkImport(Rule, id="SW-MIG0001", applicability="review"):
    """A Swift source explicitly imports a platform UI framework."""

    provenance = apple_review("ImportDeclSyntax: UIKit/AppKit named imports")
    query = """
        SELECT *, name AS framework,
               'https://developer.apple.com/documentation/swiftui/' ||
               CASE WHEN split_part(name, '.', 1) = 'AppKit' THEN 'appkit-integration'
                    ELSE 'uikit-integration' END AS reference
        FROM swift_source_imports
        WHERE split_part(name, '.', 1) IN ('UIKit', 'AppKit')
    """
    message = (
        "imports {framework}: review whether shared SwiftUI or a small platform adapter can "
        "own this dependency; imports alone do not prove a UIKit/AppKit screen. "
        "{reference}"
    )
    guidance = (
        "Inventory the actual imported APIs before moving code. Use shared SwiftUI for layout, "
        "navigation and controls where it preserves behavior. Keep a small platform adapter "
        "for services or input behavior that SwiftUI does not expose. A UIKit/AppKit import "
        "may support clipboard, lifecycle or input services without owning a screen. Configure "
        "an explicit per-rule path exclusion for a reviewed adapter. No automatic rewrite."
    )
    references = (
        "https://developer.apple.com/documentation/swiftui/uikit-integration",
        "https://developer.apple.com/documentation/swiftui/appkit-integration",
    )
    examples = (
        Example(bad="import UIKit\n", good="import SwiftUI\n"),
        Example(bad="import AppKit\n", good="import SwiftUI\n"),
    )


class PlatformUIBoundary(Rule, id="SW-MIG0002", applicability="review"):
    """A named platform view superclass or SwiftUI representable marks a UI boundary."""

    provenance = apple_review("InheritanceClauseSyntax and SwiftUI platform representables")
    query = """
        SELECT *, CASE
          WHEN contains(inherited, 'Representable') THEN
            'https://developer.apple.com/documentation/swiftui/' ||
              lower(split_part(inherited, '.', -1))
          WHEN starts_with(split_part(inherited, '.', -1), 'NS') THEN
            'https://developer.apple.com/documentation/swiftui/appkit-integration'
          ELSE 'https://developer.apple.com/documentation/swiftui/uikit-integration'
          END AS reference
        FROM swift_source_conformances
        WHERE shadowed = 0 AND split_part(inherited, '.', -1) IN (
          'UIView', 'UIViewController', 'UITableViewController', 'UICollectionViewController',
          'UINavigationController', 'UITabBarController', 'UIHostingController',
          'NSView', 'NSViewController', 'NSWindowController', 'NSHostingController',
          'UIViewRepresentable', 'UIViewControllerRepresentable',
          'NSViewRepresentable', 'NSViewControllerRepresentable')
          AND (NOT contains(inherited, '.') OR inherited IN (
            'UIKit.UIView', 'UIKit.UIViewController', 'UIKit.UITableViewController',
            'UIKit.UICollectionViewController', 'UIKit.UINavigationController',
            'UIKit.UITabBarController', 'AppKit.NSView', 'AppKit.NSViewController',
            'AppKit.NSWindowController', 'SwiftUI.UIHostingController',
            'SwiftUI.NSHostingController', 'SwiftUI.UIViewRepresentable',
            'SwiftUI.UIViewControllerRepresentable', 'SwiftUI.NSViewRepresentable',
            'SwiftUI.NSViewControllerRepresentable'))
    """
    message = (
        "`{owner}` inherits/conforms to named type `{inherited}`: review replacing the screen "
        "with shared SwiftUI or documenting the smallest required bridge. "
        "Syntax names are not resolved types. {reference}"
    )
    guidance = (
        "First identify behavior SwiftUI can preserve directly; replace that UI incrementally "
        "and test it. Keep a representable when platform functionality is still required, "
        "with lifecycle and delegate coordination in that bridge. This audit reports named "
        "inheritance, including conditional branches, excluding known project shadows and "
        "non-framework qualifiers. It cannot resolve aliases, external same-named types or "
        "subclasses through another superclass. An intentional bridge is "
        "a supported Apple integration mechanism, not an error or an automatic deletion."
    )
    references = (
        "https://developer.apple.com/documentation/swiftui/uikit-integration",
        "https://developer.apple.com/documentation/swiftui/uiviewrepresentable",
        "https://developer.apple.com/documentation/swiftui/nsviewrepresentable",
    )
    examples = (
        Example(
            bad="import SwiftUI\nstruct Input: UIViewRepresentable {}\n",
            good="import SwiftUI\nstruct Input: View {}\n",
        ),
    )


class ObjectiveCExposure(Rule, id="SW-MIG0003", applicability="review"):
    """Explicit Objective-C exposure remains in Swift source."""

    provenance = apple_review("AttributeSyntax: explicit objc/objcMembers runtime exposure")
    query = """
        SELECT * FROM swift_source_attributes WHERE attribute IN ('objc', 'objcMembers')
    """
    message = (
        "`{owner}` uses @{attribute}: review selector, delegate and Objective-C callers before "
        "removing this interoperability boundary. "
        "https://developer.apple.com/documentation/swift/using-objective-c-runtime-features-in-swift"
    )
    guidance = (
        "Find Objective-C, selector, delegate, KVC/KVO and framework callers first. Prefer "
        "Swift closures or SwiftUI actions for app-owned UI when equivalent behavior exists, "
        "but retain @objc where a framework or caller requires runtime dispatch. This is a "
        "factual migration inventory, not evidence that the attribute is unnecessary. No "
        "automatic removal is safe from syntax alone."
    )
    references = (
        "https://developer.apple.com/documentation/swift/using-objective-c-runtime-features-in-swift",
        "https://developer.apple.com/documentation/swift/migrating-your-objective-c-code-to-swift",
    )
    examples = (Example(bad="@objc class Model {}\n", good="class Model {}\n"),)


class HandwrittenDatabaseDependency(Rule, id="SW-MIG0004", applicability="review"):
    """A Swift source explicitly imports a direct SQLite interface."""

    provenance = apple_review("ImportDeclSyntax: SQLite3/FMDB; SwiftData public model/query APIs")
    query = """
        SELECT *, name AS framework FROM swift_source_imports
        WHERE split_part(name, '.', 1) IN ('SQLite3', 'FMDB')
    """
    message = (
        "imports {framework}: review replacing app-owned SQL persistence with SwiftData "
        "models and queries; preserve migration, transactions and offline data first. "
        "Imports do not prove SQL execution. https://developer.apple.com/documentation/swiftdata"
    )
    guidance = (
        "Trace app-owned queries, schema and migration duties before replacing the interface. "
        "Use SwiftData models, FetchDescriptor/Query and ModelContext where they preserve "
        "account isolation, transactions and offline behavior. Keep any required read-only "
        "legacy importer isolated until migration is verified. This reports SQLite3/FMDB "
        "import syntax only; it neither searches string literals nor proves SQL execution. "
        "CoreData imports are not treated as handwritten SQL. No automatic schema rewrite."
    )
    references = ("https://developer.apple.com/documentation/swiftdata",)
    examples = (Example(bad="import SQLite3\n", good="import SwiftData\n"),)
