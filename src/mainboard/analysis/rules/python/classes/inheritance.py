"""Rules over how classes inherit (mcmr's PY-CLAS inheritance family)."""

from ....catalog import Rule


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


class PassThroughInheritanceLayerCount(Rule, id="PY-CLAS0006"):
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


class HazardousMultipleInheritanceMroCount(Rule, id="PY-CLAS0007"):
    """Deterministic MRO hazards among project-owned direct bases."""

    query = """
            SELECT c.*, b.bases FROM class_classes c
            JOIN (SELECT class_id, string_agg(value, '`, `' ORDER BY ordinal) AS bases
                  FROM class_direct_bases GROUP BY class_id HAVING count(*) >= 2) b
              USING (class_id)
            WHERE (c.has_redundant_direct_base OR c.has_noncooperative_concrete_collision)
              AND NOT EXISTS (SELECT 1 FROM class_class_decorators d WHERE d.class_id = c.class_id)
        """
    message = (
        "`{name}` combines `{bases}` with a redundant base or a noncooperative concrete collision"
    )
