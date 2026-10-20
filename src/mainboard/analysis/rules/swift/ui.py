"""SwiftUI ownership and avoidable type erasure, grounded in syntax rather than source regexes."""

from ...catalog import Example, Rule


class PrivateViewState(Rule, id="SW-UI0001"):
    """SwiftUI-owned state exposed through a view's access level."""

    query = "SELECT * FROM swift_source_states WHERE is_private = 0"
    message = (
        "`{name}` uses @{attribute}; keep SwiftUI-owned state private and pass Binding to "
        "children. Review initializer/caller compatibility before changing access. "
        "https://developer.apple.com/documentation/swiftui/state"
    )
    guidance = (
        "Declare @State and @StateObject storage private so a memberwise initializer cannot "
        "replace SwiftUI-managed storage. Pass values or Binding to child views. Review custom "
        "initializers and external callers first; no automatic access-control change is safe "
        "without resolving those uses. This syntax check recognizes direct View conformances "
        "with import SwiftUI, not inherited conformances or resolved types."
    )
    references = (
        "https://developer.apple.com/documentation/swiftui/state",
        "https://developer.apple.com/documentation/swiftui/stateobject",
    )
    examples = (
        Example(
            bad="import SwiftUI\nstruct Screen: View { @State var selected = false }\n",
            good="import SwiftUI\nstruct Screen: View { @State private var selected = false }\n",
        ),
    )


class RedundantBodyErasure(Rule, id="SW-UI0002", applicability="review"):
    """A sole AnyView wrapper in an opaque SwiftUI body can hide its concrete type."""

    query = "SELECT * FROM swift_source_erasures"
    message = (
        "`{view_name}.body` only wraps its result in AnyView; consider returning the concrete "
        "view from some View. Keep type erasure when the API requires it. "
        "https://developer.apple.com/documentation/swiftui/anyview"
    )
    guidance = (
        "For a single direct AnyView expression in body: some View, review returning the "
        "wrapped view directly so SwiftUI retains its concrete structure. This is a migration "
        "review suggestion, not a ban on AnyView. Collections, dynamically selected content, "
        "branches, helper methods and bodies explicitly returning AnyView are left alone. "
        "The parser does not resolve types, overloads or macros, so no rewrite is offered."
    )
    references = ("https://developer.apple.com/documentation/swiftui/anyview",)
    examples = (
        Example(
            bad='import SwiftUI\nstruct Screen: View { var body: some View { AnyView(Text("Hi")) } }\n',
            good='import SwiftUI\nstruct Screen: View { var body: some View { Text("Hi") } }\n',
        ),
    )
