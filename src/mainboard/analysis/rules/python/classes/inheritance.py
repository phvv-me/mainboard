"""Rules over how classes inherit (mcmr's PY-CLAS inheritance family)."""

from ....catalog import Example, Rule

# A base, a layer that adds nothing and a subclass of the layer.
_STACK = """class Base:
    def run(self):
        return 1


class Mid(Base):
    pass


class Leaf(Mid):
    def run(self):
        return 2
"""
# A base, its subclass, an unrelated class and a class whose bases the example states.
_BASES = """class A:
    def run(self):
        return 1


class B(A):
    def run(self):
        return 2


class E:
    pass


class C({}):
    pass
"""
# Two bases that define the same method and neither delegates to the other.
_COLLISION = """class AppleSilicon:
    def label(self):
        return "apple"


class GPU:
    def label(self):
        return "gpu"


class AppleGPU(AppleSilicon, GPU):
    pass
"""


class ArtificialSingleSubclassBaseCount(Rule, id="PY-CLAS0005"):
    """Concrete project bases that exist only for one closed-world subclass."""

    query = """
            SELECT c.*, s.subclass FROM class_classes c
            JOIN (SELECT class_id, min(value) AS subclass FROM class_direct_subclasses
                  GROUP BY class_id HAVING count(*) = 1) s USING (class_id)
            WHERE c.scope = 'module' AND c.descendant_count = 1 AND NOT c.is_instantiated
              AND NOT c.is_exported AND c.only_cross_module_reference_is_subclass
              AND NOT EXISTS (SELECT 1 FROM class_class_decorators d WHERE d.class_id = c.class_id)
              AND NOT EXISTS (SELECT 1 FROM class_class_keywords k WHERE k.class_id = c.class_id)
              AND NOT EXISTS (SELECT 1 FROM class_direct_bases b
                              WHERE b.class_id = c.class_id AND b.value <> 'object')
        """
    message = "`{name}` exists only as the base of `{subclass}`"


class PassThroughInheritanceLayerCount(Rule, id="PY-CLAS0006", applicability="review"):
    """Project-owned inheritance layers that add only a name or forwarding frame. A class under
    `if TYPE_CHECKING:` is a declaration for the type checker, where a bare subclass is a distinct
    type (`class u8x4(Packed): ...`)."""

    contract_suffixes: tuple[str, ...] = (
        "Backend",
        "Error",
        "Exception",
        "Mixin",
        "Plugin",
        "Port",
        "Protocol",
        "Provider",
        "Registry",
        "Strategy",
    )

    query = """
            SELECT c.*, b.base FROM class_classes c
            JOIN (SELECT class_id, min(value) AS base FROM class_direct_bases
                  GROUP BY class_id HAVING count(*) = 1) b USING (class_id)
            WHERE c.scope = 'module' AND c.is_pass_through_layer
              AND NOT c.base_is_removable_overlap
              AND NOT EXISTS (SELECT 1 FROM unnest($contract_suffixes::VARCHAR[]) AS s(suffix)
                              WHERE ends_with(c.name, s.suffix))
              AND NOT EXISTS (SELECT 1 FROM class_class_decorators d WHERE d.class_id = c.class_id)
              AND NOT EXISTS (SELECT 1 FROM type_checking_blocks() t
                              WHERE t.path = c.path AND c.start_line BETWEEN t.start_line AND t.end_line)
        """
    message = "`{name}` passes through its only base `{base}`"
    # The layer is deleted and each class that has it for its only base takes the layer's base
    # instead, when every spelling of the layer's name in the project is one of those bases or
    # its definition. A layer anything else names (an import, an annotation, a call, a string) is
    # left to a person, as is a layer that is exported, instantiated or takes keywords.
    fix = r"""
        WITH layers AS (
          SELECT f.finding_key, f.path, f.name, f.base, f.class_id, d.class_node
          FROM finding f JOIN class_defs d ON d.class_id = f.class_id
          WHERE NOT f.is_exported AND NOT f.is_instantiated
            AND NOT EXISTS (SELECT 1 FROM class_header_items k
                            WHERE k.class_id = f.class_id AND k.is_keyword)
        ), bases AS (
          SELECT l.finding_key, h.start_line, h.start_column, h.end_line, h.end_column
          FROM layers l
          JOIN class_header_items h ON h.path = l.path AND h.text = l.name
                                   AND h.class_id <> l.class_id AND NOT h.is_keyword
          WHERE (SELECT count(*) FROM class_header_items o
                 WHERE o.class_id = h.class_id AND NOT o.is_keyword) = 1
        ), spelled AS (
          SELECT l.finding_key,
                 sum(len(regexp_extract_all(s.content, '\b' || l.name || '\b'))) AS spellings
          FROM layers l JOIN source_files s ON contains(s.content, l.name)
          GROUP BY l.finding_key
        ), removable AS (
          SELECT l.* FROM layers l JOIN spelled s ON s.finding_key = l.finding_key
          WHERE s.spellings = 1 + (SELECT count(*) FROM bases b WHERE b.finding_key = l.finding_key)
        )
        SELECT r.finding_key, 'delete-node' AS operation, r.path, r.class_node AS node,
               NULL AS start_line, NULL AS start_column, NULL AS end_line, NULL AS end_column,
               NULL AS text
        FROM removable r
        UNION ALL
        SELECT r.finding_key, NULL, r.path, NULL, b.start_line, b.start_column, b.end_line,
               b.end_column, r.base
        FROM bases b JOIN removable r ON r.finding_key = b.finding_key
    """
    change = "deleted the layer `{name}`, its subclasses now inherit `{base}`"
    examples = (
        Example(
            bad=_STACK,
            good="""class Base:
    def run(self):
        return 1


class Leaf(Base):
    def run(self):
        return 2
""",
        ),
        Example(
            bad="""class Base:
    def __init__(self, size):
        self.size = size


class Mid(Base):
    def __init__(self, size):
        return super().__init__(size)


class Leaf(Mid):
    def grow(self):
        return self.size + 1
""",
            good="""class Base:
    def __init__(self, size):
        self.size = size


class Leaf(Base):
    def grow(self):
        return self.size + 1
""",
        ),
        Example(
            bad=_STACK,
            good=_STACK,
            context={"other.py": "from module import Mid\n\nLAYER = Mid\n"},
        ),
    )


class HazardousMultipleInheritanceMroCount(Rule, id="PY-CLAS0007", applicability="review"):
    """Deterministic MRO hazards among project-owned direct bases."""

    # A redundant base shares its methods with the base that inherits it, which the collision half
    # counts too, so the redundancy is named first.
    query = """
            SELECT c.*, b.bases,
                   CASE WHEN c.has_redundant_direct_base THEN 'a redundant base'
                        ELSE 'a noncooperative concrete collision' END AS hazard
            FROM class_classes c
            JOIN (SELECT class_id, string_agg(value, '`, `' ORDER BY ordinal) AS bases
                  FROM class_direct_bases GROUP BY class_id HAVING count(*) >= 2) b
              USING (class_id)
            WHERE (c.has_redundant_direct_base OR c.has_noncooperative_concrete_collision)
              AND NOT EXISTS (SELECT 1 FROM class_class_decorators d WHERE d.class_id = c.class_id)
        """
    message = "`{name}` combines `{bases}` with {hazard}"
    # Only the redundant half has a fix: the last base goes when the base before it already
    # inherits it, which keeps the class's method resolution order. A redundant base elsewhere
    # would change it (`C(B, E, A)` resolves `E` before `A`, `C(B, E)` after it). A class with a
    # second redundant base keeps both, since dropping one leaves its finding standing and the
    # fixpoint withholds a fix that clears none. A collision is a design choice, left to a person.
    fix = """
        WITH redundant AS (
          SELECT DISTINCT e.child AS class_id, e.parent, e.ordinal
          FROM class_edges e
          JOIN class_edges o ON o.child = e.child AND o.parent <> e.parent
          JOIN class_lineage l ON l.class_id = o.parent AND l.ancestor = e.parent
        )
        SELECT DISTINCT f.finding_key, 'drop-argument' AS operation, f.path, h.node_id AS node
        FROM finding f
        JOIN redundant r ON r.class_id = f.class_id
        JOIN class_edges p ON p.child = r.class_id AND p.ordinal = r.ordinal - 1
        JOIN class_lineage l ON l.class_id = p.parent AND l.ancestor = r.parent
        JOIN class_header_items h ON h.class_id = r.class_id AND NOT h.is_keyword
                                 AND h.ordinal = r.ordinal
        WHERE r.ordinal = (SELECT max(o.ordinal) FROM class_header_items o
                           WHERE o.class_id = r.class_id AND NOT o.is_keyword)
          AND NOT EXISTS (SELECT 1 FROM redundant x
                          WHERE x.class_id = r.class_id AND x.ordinal <> r.ordinal)
    """
    change = "dropped the redundant last base of `{name}`"
    examples = (
        Example(bad=_BASES.format("B, A"), good=_BASES.format("B")),
        Example(
            bad=_BASES.format("B, A, metaclass=type"), good=_BASES.format("B, metaclass=type")
        ),
        Example(bad=_COLLISION, good=_COLLISION),
        Example(bad=_BASES.format("B, E, A"), good=_BASES.format("B, E, A")),
    )
