"""Rules over how data models are declared and built (mcmr's PY-MODE family)."""

from ...catalog import Example, Rule


class SharedModelFileShape(Rule, id="PY-MODE0001"):
    """Shared `models` package files that do not hold exactly one model-shaped class."""

    query = """
        SELECT * FROM class_model_files
        WHERE NOT is_package_initializer AND (top_level_class_count <> 1 OR model_class_count <> 1)
    """
    message = (
        "`{path}` declares {top_level_class_count} top-level classes of which "
        "{model_class_count} are models"
    )


class ApprovedModelFoundation(Rule, id="PY-MODE0003", applicability="review"):
    """Direct Pydantic models that bypass an established approved house base."""

    approved_base: str = "patos.FrozenModel"

    query = """
        SELECT c.* FROM class_classes c JOIN class_facts f USING (fact_id)
        WHERE f.has_approved_model_foundation_policy AND c.directly_inherits_pydantic_base_model
          AND NOT c.inherits_approved_model_foundation
    """
    message = "`{name}` inherits Pydantic directly instead of the approved model foundation"
    examples = (
        Example(
            bad="from patos import FrozenModel\nfrom pydantic import BaseModel\n\n\n"
            "class Config(FrozenModel):\n    size: int\n\n\n"
            "class User(BaseModel):\n    name: str\n",
            good="from patos import FrozenModel\nfrom pydantic import BaseModel\n\n\n"
            "class Config(FrozenModel):\n    size: int\n\n\n"
            "class User(FrozenModel):\n    name: str\n",
        ),
    )
    # The `BaseModel` in the class header becomes the approved base, imported unless the file
    # binds that name already. The base is immutable (`approved_base`): a class that assigns to
    # its fields needs a person to pick the mutable one. A header that is not on one line, or
    # names the base twice, is left.
    fix = """
        WITH pydantic AS (
          SELECT path, name AS spelled FROM import_binding_facts
          WHERE module = 'pydantic' AND imported_name = 'BaseModel'
          UNION ALL
          SELECT path, name || '.BaseModel' FROM import_binding_facts
          WHERE module = 'pydantic' AND imported_name = 'pydantic'
        ), located AS (
          SELECT f.finding_key, f.path, f.start_line, p.spelled, h.text AS header,
                 strpos(h.text, '(') AS opened
          FROM finding f
          JOIN class_direct_bases b ON b.class_id = f.class_id
          JOIN pydantic p ON p.path = f.path AND p.spelled = b.value
          JOIN imports_code_lines() h ON h.path = f.path AND h.line = f.start_line
          WHERE regexp_matches(h.text, '^\\s*class\\s')
        ), placed AS (
          SELECT *, strlen(left(header,
                      opened + strpos(substring(header, opened + 1), spelled) - 1)) AS base_column
          FROM located
          WHERE opened > 0 AND len(string_split(substring(header, opened + 1), spelled)) = 2
        )
        SELECT finding_key, path, start_line, base_column AS start_column,
               start_line AS end_line, base_column + strlen(spelled) AS end_column,
               regexp_extract($approved_base, '[^.]+$') AS text,
               CASE WHEN NOT EXISTS (SELECT 1 FROM import_binding_facts i
                                     WHERE i.path = placed.path
                                       AND i.name = regexp_extract($approved_base, '[^.]+$'))
                    THEN 'from ' || regexp_replace($approved_base, '\\.[^.]+$', '')
                         || ' import ' || regexp_extract($approved_base, '[^.]+$') END
                 AS "import"
        FROM placed
    """


class ManualModelAttributeProjectionCount(Rule, id="PY-MODE0004"):
    """Structures that manually repeat fields from one model instance."""

    minimum_attributes: int = 4

    query = """
        WITH attributes AS (SELECT DISTINCT projection_id, value FROM class_projection_attributes),
        outputs AS (
          SELECT DISTINCT projection_id, replace(value, '-', '_') AS value
          FROM class_projection_output_keys),
        sizes AS (
          SELECT projection_id, count(*) AS attribute_count, count(o.value) AS shared_count
          FROM attributes a LEFT JOIN outputs o USING (projection_id, value)
          GROUP BY projection_id),
        written AS (
          SELECT projection_id, count(*) AS output_count FROM outputs GROUP BY projection_id),
        names AS (
          SELECT projection_id, string_agg(value, '`, `' ORDER BY ordinal) AS names
          FROM class_projection_attributes GROUP BY projection_id)
        SELECT p.*, n.names FROM class_projections p
        LEFT JOIN sizes s USING (projection_id)
        LEFT JOIN written w USING (projection_id)
        LEFT JOIN names n USING (projection_id)
        WHERE coalesce(s.attribute_count, 0) >= $minimum_attributes
          AND coalesce(s.attribute_count, 0) = coalesce(w.output_count, 0)
          AND coalesce(s.attribute_count, 0) = coalesce(s.shared_count, 0)
    """
    message = "projection from `{root}` repeats model attributes `{names}`"


class StandardDataclassModel(Rule, id="PY-MODE0005"):
    """State-bearing standard dataclasses that should use the approved model foundation."""

    query = """
        SELECT * FROM class_classes
        WHERE is_dataclass AND field_count > 0 AND NOT has_ordinary_behavior
    """
    message = (
        "`{name}` is a standard dataclass with {field_count} fields instead of an approved "
        "Pydantic model"
    )


class EmptyDeclarativeModel(Rule, id="PY-MODE0006"):
    """Declarative models that own no field. A class body that assigns a call to a lowercase name
    declares its fields in a column DSL (`name = sql.PK(str)`, and so do the classes derived from
    it), a model naming a `metaclass` is the base that DSL's models derive from, and a registry
    implementation (`name = "leech"`) is keyed, not a record."""

    query = r"""
        SELECT * FROM class_classes c
        WHERE is_declarative_model AND field_count = 0 AND NOT has_inherited_fields
          AND NOT is_protocol
          AND NOT has_explicit_registry_name
          AND NOT EXISTS (SELECT 1 FROM class_defs d
                          JOIN class_body_assigns a ON a.fact_id = d.fact_id
                            AND a.class_node = d.class_node
                          WHERE (d.class_id = c.class_id
                                 OR d.class_id IN (SELECT ancestor FROM class_lineage l
                                                   WHERE l.class_id = c.class_id))
                            AND a.annotation IS NULL
                            AND a.value_type = 'call' AND regexp_matches(a.target_name, '\p{Ll}'))
          AND NOT EXISTS (SELECT 1 FROM class_class_keywords k
                          WHERE k.class_id = c.class_id AND starts_with(k.value, 'metaclass='))
          AND NOT EXISTS (SELECT 1 FROM class_direct_bases b WHERE b.class_id = c.class_id
                          AND regexp_replace(b.value, '^.*\.', '', 'g')
                              IN ('ABC', 'Protocol', 'ABCMeta'))
          AND NOT EXISTS (SELECT 1 FROM class_methods m
                          JOIN class_method_decorators d USING (method_id)
                          WHERE m.class_id = c.class_id
                            AND regexp_replace(d.value, '^.*\.', '', 'g')
                                IN ('abstractmethod', 'abstractproperty'))
    """
    message = "`{name}` is a model with no fields"
