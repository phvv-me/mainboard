"""A finite UIKit syntax inventory, including transitive references without UIKit imports."""

from ...catalog import Example, Rule
from .provenance import apple_review


class UIKitSymbolReference(Rule, id="SW-MIG0005", applicability="review"):
    """A directly named UIKit presentation or layout type remains in Swift source."""

    provenance = apple_review(
        "Finite UIKit name/receiver syntax and SwiftUI presentation alternatives"
    )
    query = """
        SELECT r.path, r.start_line, r.start_column, r.name, ui.alternative
        FROM swift_source_references r
        JOIN (VALUES
            ('UIView', 'shared SwiftUI View content'),
            ('UIViewController', 'SwiftUI View content and navigation'),
            ('UIButton', 'Button'),
            ('UILabel', 'Text'),
            ('UITextField', 'TextField or SecureField where input behavior is equivalent'),
            ('UIImage', 'Image for presentation; retain required image processing'),
            ('UIImageView', 'Image'),
            ('UIColor', 'Color while preserving semantic colors and the existing palette'),
            ('UIFont', 'Font while preserving chosen typography and Dynamic Type'),
            ('UIScreen', 'container geometry and adaptive SwiftUI layout'),
            ('UIApplication', 'App, scenePhase or openURL where the actual API permits'),
            ('UIWindow', 'Scene and WindowGroup where window behavior is equivalent'),
            ('UINavigationController', 'NavigationStack or NavigationSplitView'),
            ('UISplitViewController', 'NavigationSplitView'),
            ('UIStackView', 'HStack, VStack or lazy stacks'),
            ('UITableView', 'List or Table where row behavior is equivalent'),
            ('UITableViewController', 'List or Table and SwiftUI navigation'),
            ('UIScrollView', 'ScrollView')
        ) AS ui(name, alternative) ON ui.name = r.name
        WHERE r.receiver IN ('', 'UIKit') AND r.shadowed = 0
    """
    message = (
        "UIKit-named `{name}` reference: review {alternative}. Keep required input, "
        "image-processing and system bridges. Syntax names are not resolved types. "
        "https://developer.apple.com/documentation/swiftui/uikit-integration"
    )
    guidance = (
        "Review the reported constructor, type annotation or member-base reference before "
        "moving app presentation and layout to shared SwiftUI. Keep existing study behavior, "
        "visual identity, accessibility and Japanese text handling. UIKit representables are "
        "supported Apple integration points: preserve necessary IME, marked-text/caret, image "
        "processing and framework services behind the smallest useful boundary. Image can "
        "present a UIImage without replacing its processing pipeline. These are alternatives "
        "to assess, not equivalent API substitutions. This finite named-symbol inventory "
        "includes bare references without an explicit UIKit import because another module may "
        "re-export UIKit. It excludes known project shadows and non-UIKit receivers, but does "
        "not resolve external symbols, aliases or macro-generated code. It does not match UI "
        "or NS name prefixes. Review intentional bridges by path; no automatic rewrite."
    )
    references = (
        "https://developer.apple.com/documentation/swiftui/uikit-integration",
        "https://developer.apple.com/documentation/swiftui/layout-fundamentals",
        "https://developer.apple.com/documentation/swiftui/image",
        "https://developer.apple.com/documentation/swiftui/migrating-to-new-navigation-types",
    )
    examples = (
        Example(
            bad='let image = UIImage(named: "radical")\n',
            good='import SwiftUI\nlet image = Image("radical")\n',
        ),
    )
