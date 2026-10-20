"""Placed persistence syntax for review, without inferring model ownership or writer races."""

from ...catalog import Example, Rule
from .provenance import apple_review


class OpaqueModelAttribute(Rule, id="SW-MIG0006", applicability="review"):
    """A SwiftData model property explicitly selects Codable or transformable storage."""

    provenance = apple_review("SwiftData.Schema.Attribute.Option: codable/transformable(by:)")
    query = """
        SELECT o.path, o.start_line, o.start_column, o.container, o.owner, o.option
        FROM swift_source_attribute_options o
        WHERE o.container_kind = 'class' AND o.is_static = 0
          AND o.attribute IN ('Attribute', 'SwiftData.Attribute')
          AND o.attribute_shadowed = 0 AND o.shadowed = 0
          AND o.option IN ('codable', 'transformable')
          AND o.receiver IN ('', 'Schema.Attribute.Option', 'SwiftData.Schema.Attribute.Option')
          AND EXISTS (SELECT 1 FROM swift_source_imports i
              WHERE i.fact_id = o.fact_id AND split_part(i.name, '.', 1) = 'SwiftData')
          AND EXISTS (SELECT 1 FROM swift_source_attributes a
              WHERE a.fact_id = o.fact_id AND a.owner_id = o.container_id
                AND a.attribute IN ('Model', 'SwiftData.Model') AND a.shadowed = 0)
    """
    message = (
        "`{container}.{owner}` selects .{option} storage: review supported model/value types "
        "for data you own, query requirements and compatible payload decoding before changing "
        "the stored format. Syntax does not establish type ownership or an invalid model. "
        "https://developer.apple.com/videos/play/wwdc2026/274/"
    )
    guidance = (
        "Apple supports .codable (27+) and transformable attributes for encoded values. Their "
        "encoded contents are opaque to predicates and sorting, and a Codable payload shape "
        "change does not itself trigger a schema migration. Review native supported values or "
        "models for app-owned data; retain justified framework types, immutable request and "
        "recovery envelopes when necessary. Preserve old decoding, identity, transaction and "
        "recovery behavior and prove populated-store upgrade/reopen before deleting a stored "
        "representation. This audit only identifies direct options on an instance property "
        "of a directly @Model-annotated class importing SwiftData, excluding known project "
        "shadows. It cannot decide type ownership, query needs, migration compatibility or "
        "whether the encoding is unnecessary. Aliased/computed options, external macro "
        "expansions and Core Data model files are outside this syntax inventory. No rewrite."
    )
    references = (
        "https://developer.apple.com/videos/play/wwdc2026/274/",
        "https://developer.apple.com/documentation/swiftdata/schema/attribute/option/codable",
        "https://developer.apple.com/documentation/swiftdata/schema/attribute/option/"
        "transformable(by:)-9d4xh",
        "https://developer.apple.com/documentation/swiftdata/schemamigrationplan",
    )
    examples = (
        Example(
            bad="import SwiftData\n@Model class Record { @Attribute(.codable) var value: Payload }\n",
            good="import SwiftData\n@Model class Record { var title: String }\n",
        ),
    )


class PersistenceConstruction(Rule, id="SW-MIG0007", applicability="review"):
    """A directly named SwiftData ModelContainer or ModelContext constructor needs an owner."""

    provenance = apple_review("SwiftData.ModelContainer/ModelContext/ModelActor declarations")
    query = """
        SELECT c.path, c.start_line, c.start_column,
               CASE WHEN c.name = 'init' THEN split_part(c.receiver, '.', -1)
                    ELSE c.name END AS type_name
        FROM swift_source_calls c
        WHERE c.shadowed = 0
          AND ((c.name IN ('ModelContainer', 'ModelContext') AND c.receiver IN ('', 'SwiftData'))
            OR (c.name = 'init' AND c.receiver IN (
                'ModelContainer', 'ModelContext', 'SwiftData.ModelContainer',
                'SwiftData.ModelContext')))
          AND EXISTS (SELECT 1 FROM swift_source_imports i
              WHERE i.fact_id = c.fact_id AND split_part(i.name, '.', 1) = 'SwiftData')
    """
    message = (
        "`{type_name}` construction: review store lifetime, actor/account ownership and "
        "injection at the composition boundary. This does not prove a duplicate writer; "
        "migration and test contexts can be legitimate. "
        "https://developer.apple.com/documentation/swiftdata/modelcontainer"
    )
    guidance = (
        "A ModelContainer coordinates a schema and storage configuration; its contexts track "
        "model lifecycles and pending changes. Inventory directly named construction sites "
        "and ensure each has a deliberate owner/lifetime. Reuse the app's established "
        "composition/store boundary instead of creating unrelated feature stores or passing "
        "live models across executors. Multiple contexts, test containers, migration stores "
        "and distinct persistent stores are supported and may be intentional. Review actual "
        "callers, isolation and account cancellation with compiler/runtime tests before "
        "consolidating them; do not create a hidden global singleton. This syntax audit "
        "requires a SwiftData import and a direct bare/module-qualified constructor or "
        "explicit .init with no known project shadow. It cannot establish store identity, "
        "actor ownership or competing writers and does not trace type aliases, inferred .init "
        "or factory calls. No automatic rewrite."
    )
    references = (
        "https://developer.apple.com/documentation/swiftdata/modelcontainer",
        "https://developer.apple.com/documentation/swiftdata/modelcontext",
        "https://developer.apple.com/documentation/swiftdata/modelactor",
    )
    examples = (
        Example(
            bad="import SwiftData\nlet context = ModelContext(container)\n",
            good="import SwiftData\nlet context = container.mainContext\n",
        ),
    )


class DirectSQLiteCall(Rule, id="SW-MIG0008", applicability="review"):
    """A finite directly named SQLite execution/preparation call remains in Swift source."""

    provenance = apple_review("SDK sqlite3.h: sqlite3_exec/prepare/prepare_v2/prepare_v3/step")
    query = """
        SELECT c.path, c.start_line, c.start_column, c.name
        FROM swift_source_calls c
        WHERE c.name IN ('sqlite3_exec', 'sqlite3_prepare', 'sqlite3_prepare_v2',
                         'sqlite3_prepare_v3', 'sqlite3_step')
          AND c.receiver IN ('', 'SQLite3') AND c.shadowed = 0
          AND EXISTS (SELECT 1 FROM swift_source_imports i
              WHERE i.fact_id = c.fact_id AND split_part(i.name, '.', 1) = 'SQLite3')
    """
    message = (
        "Direct `{name}` call: review SwiftData models/queries for normal persistence. "
        "Retain required read-only legacy conversion and disposable fixture boundaries. "
        "Syntax does not establish SQL text, execution outcome or read/write mode. "
        "https://developer.apple.com/documentation/swiftdata"
    )
    guidance = (
        "Trace the reported direct SQLite execution/preparation API and its transaction, "
        "schema, identity and recovery duties. Move normal app-owned persistence to the "
        "established SwiftData store where equivalent behavior is verified. Keep isolated "
        "read-only historical importers and disposable database fixtures until migration is "
        "accepted. This finite call inventory requires SQLite3 import syntax, a direct "
        "sqlite3_exec/prepare/prepare_v2/prepare_v3/step call and no known project shadow. "
        "It neither searches arbitrary SQL strings nor resolves external symbols, aliases, "
        "wrappers or FMDB method receivers. It cannot classify SQL as a read/write, establish "
        "a successful execution, or prove a boundary removable. No schema or query rewrite."
    )
    references = (
        "https://developer.apple.com/documentation/swiftdata",
        "https://developer.apple.com/documentation/swiftdata/modelcontext",
    )
    examples = (
        Example(
            bad='import SQLite3\nlet result = sqlite3_exec(database, "SELECT 1", nil, nil, nil)\n',
            good="import SwiftData\nlet descriptor = FetchDescriptor<Record>()\n",
        ),
    )
