"""Apple implementation files to review during an explicitly chosen Swift migration."""

from ...catalog import Example, Rule
from ..swift.provenance import apple_review


class ObjectiveCImplementation(Rule, id="ALL-MIG0001", applicability="review"):
    """An Objective-C or Objective-C++ implementation remains in the project."""

    provenance = apple_review(
        "Project .m/.mm suffix inventory only; no Objective-C AST or SDK resolution"
    )
    query = "SELECT * FROM apple_source_facts"
    message = (
        "Objective-C implementation: review migrating one class/file at a time to Swift, "
        "preserving framework interoperability and behavior. File inventory only; target "
        "membership is unknown. "
        "https://developer.apple.com/documentation/swift/migrating-your-objective-c-code-to-swift"
    )
    guidance = (
        "This reports kept .m/.mm paths, not parsed code or confirmed target membership. "
        "Migrate an app-owned class incrementally, verify callers and behavior, then remove "
        "the old implementation from target membership. Objective-C++ may expose C++ APIs "
        "that need an explicit interoperability decision. Reviewed third-party/generated "
        "sources belong in lint exclusions or linguist-vendored attributes, not bulk rewrites."
    )
    references = (
        "https://developer.apple.com/documentation/swift/migrating-your-objective-c-code-to-swift",
    )
    examples = (Example(path="Legacy.m", bad="@implementation Legacy\n@end\n", good=None),)
