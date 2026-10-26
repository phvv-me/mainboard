"""Conservative SwiftUI migration audits; syntax evidence never authorizes a rewrite."""

from ....catalog import Example, Rule
from ..provenance import apple_review


class LegacyNavigationView(Rule, id="SW-UI0003", applicability="review"):
    """A directly named SwiftUI NavigationView constructor remains in modern UI code."""

    provenance = apple_review(
        "SwiftUI.NavigationView/NavigationStack/NavigationSplitView declarations"
    )
    query = """
        SELECT c.path, c.start_line, c.start_column
        FROM swift_source_calls c
        WHERE c.name = 'NavigationView' AND c.receiver IN ('', 'SwiftUI')
          AND c.shadowed = 0
          AND EXISTS (SELECT 1 FROM swift_source_imports i
              WHERE i.fact_id = c.fact_id AND i.name = 'SwiftUI')
    """
    message = (
        "NavigationView construction: review NavigationStack for one column or "
        "NavigationSplitView for adaptive columns. Preserve selection, links and restoration; "
        "check deployment targets before migrating. "
        "https://developer.apple.com/documentation/swiftui/migrating-to-new-navigation-types"
    )
    guidance = (
        "For iOS/iPadOS 16+, macOS 13+, tvOS 16+, watchOS 9+ and visionOS 1+, Apple recommends "
        "moving NavigationView to NavigationStack or NavigationSplitView. Choose by navigation "
        "behavior and available space, not a platform-name branch. Preserve deep links, current "
        "selection, back navigation, state restoration and the existing visual identity. Keep "
        "availability fallbacks when older targets require them. This audit requires an import "
        "of SwiftUI and a directly named constructor with no known project shadow; it does not "
        "resolve external symbols, overloads, aliases or macro expansions. No automatic rewrite."
    )
    references = (
        "https://developer.apple.com/documentation/swiftui/migrating-to-new-navigation-types",
    )
    examples = (
        Example(
            bad='import SwiftUI\nlet content = NavigationView { Text("Reviews") }\n',
            good='import SwiftUI\nlet content = NavigationStack { Text("Reviews") }\n',
        ),
    )


class ObservableModelMigration(Rule, id="SW-UI0004", applicability="review"):
    """A SwiftUI-importing class combines ObservableObject and instance Published storage."""

    provenance = apple_review(
        "Combine.ObservableObject/Published; Observation macro migration guidance"
    )
    query = """
        SELECT c.path, c.start_line, c.start_column, c.owner
        FROM swift_source_conformances c
        WHERE c.owner_kind = 'class' AND c.shadowed = 0
          AND c.inherited IN ('ObservableObject', 'Combine.ObservableObject',
                              'SwiftUI.ObservableObject')
          AND EXISTS (SELECT 1 FROM swift_source_imports i
              WHERE i.fact_id = c.fact_id AND i.name = 'SwiftUI')
          AND EXISTS (SELECT 1 FROM swift_source_attributes a
              WHERE a.fact_id = c.fact_id AND a.container_id = c.owner_id
                AND a.container_kind = 'class' AND a.is_static = 0 AND a.shadowed = 0
                AND a.attribute IN ('Published', 'Combine.Published', 'SwiftUI.Published'))
          AND NOT EXISTS (SELECT 1 FROM swift_source_attributes a
              WHERE a.fact_id = c.fact_id AND a.owner_id = c.owner_id
                AND a.shadowed = 0
                AND a.attribute IN ('Observable', 'Observation.Observable', 'SwiftUI.Observable'))
    """
    message = (
        "`{owner}` combines ObservableObject and @Published: review @Observable for "
        "SwiftUI-owned models. Preserve Combine publishers and external protocol requirements; "
        "this is a migration audit, not a ban. "
        "https://developer.apple.com/documentation/swiftui/"
        "migrating-from-the-observable-object-protocol-to-the-observable-macro"
    )
    guidance = (
        "SwiftUI supports Observation on iOS/iPadOS 17+, macOS 14+, tvOS 17+ and watchOS 10+. "
        "For app-owned models, review replacing ObservableObject and @Published with "
        "@Observable, then migrate owning StateObject/ObservedObject/EnvironmentObject uses "
        "and binding access deliberately. Observation tracks properties read by a view, so "
        "updates can differ from objectWillChange. Retain the existing protocol and publishers "
        "when external APIs, subscribers or older deployment targets require them. Apple "
        "supports incremental migration; a wholesale replacement is unnecessary. This syntax "
        "audit requires import SwiftUI, a direct class conformance and a nonstatic @Published "
        "member with no known project shadows. It cannot prove app ownership, resolve external "
        "requirements or infer subscriber use. Combine-only files are left alone. No rewrite."
    )
    references = (
        "https://developer.apple.com/documentation/swiftui/"
        "migrating-from-the-observable-object-protocol-to-the-observable-macro",
        "https://developer.apple.com/documentation/observation",
    )
    examples = (
        Example(
            bad=(
                """import SwiftUI
final class ReviewModel: ObservableObject { @Published var answer = "" }
"""
            ),
            good=('import SwiftUI\n@Observable final class ReviewModel { var answer = "" }\n'),
        ),
    )
