"""Rules over how Pydantic models are built (mcmr's PY-PYDA construction family)."""

from keyword import kwlist

from ....catalog import Example, Rule

# mcmr's `^[^\W\d]\w*$` under Rust's Unicode `\w`, which RE2 limits to ASCII.
_IDENTIFIER = r"^[\p{L}\p{Nl}\p{M}\p{Pc}][\p{L}\p{Nl}\p{M}\p{Nd}\p{Pc}]*$"


class RedundantModelValidate(Rule, id="PY-PYDA0004", applicability="safe"):
    """`model_validate` calls that already spell out ordinary constructor fields.

    Only string keys spell a field: a key held in a variable (`{_PYTHON: ...}`) names whatever
    the variable holds, which a keyword argument would replace with the variable's own name.
    Only literal and named values pass a field as it is typed: a value some call computes (raw
    JSON, `json.loads(...)["x"]`) is what `model_validate` exists to check, and as a keyword the
    type checkers refuse it, since they can prove nothing about it.
    """

    query = """
        WITH calls AS (
          SELECT c.* FROM call_calls c JOIN call_facts f USING (fact_id)
          WHERE ends_with(c.qualified_name, '.model_validate') AND NOT f.is_test
            AND NOT EXISTS (SELECT 1 FROM call_keywords k WHERE k.call_id = c.call_id)
        ), arguments AS (
          SELECT e.* FROM call_expressions e SEMI JOIN calls USING (call_id)
          WHERE e.root_relation = 'argument' AND e.depth = 0
        ), mappings AS (
          SELECT m.expression_id AS mapping_id,
            string_agg('`' || m.key || '`', ', ' ORDER BY m.ordinal) AS fields,
            string_agg(m.key || '=' || v.text, ', ' ORDER BY m.ordinal) AS keywords,
            bool_and(m.is_string_key AND regexp_matches(m.key, $identifier)
                     AND NOT list_contains($keywords::VARCHAR[], m.key)
                     AND (v.literal_kind <> 'none'
                          OR regexp_matches(v.text, '^[A-Za-z_][A-Za-z0-9_.]*$')))
              AS states_only_fields,
            count(*) AS entry_count
          FROM call_mapping_entries m
          JOIN call_expressions v ON v.expression_id = m.value_expression_id
          WHERE m.expression_id IN (SELECT expression_id FROM arguments)
          GROUP BY m.expression_id
        )
        SELECT c.*, c.node_path AS path, c.node_start_line AS start_line,
               c.node_start_column AS start_column, c.node_end_line AS end_line,
               c.node_end_column AS end_column, m.fields, m.keywords, r.text AS receiver
        FROM calls c
        JOIN (SELECT call_id, count(*) AS argument_count FROM arguments GROUP BY call_id) n
          USING (call_id)
        JOIN arguments a ON a.call_id = c.call_id AND a.root_ordinal = 0
        JOIN mappings m ON m.mapping_id = a.expression_id
        LEFT JOIN call_expressions r ON r.call_id = c.call_id
          AND r.root_relation = 'receiver' AND r.depth = 0
        WHERE n.argument_count = 1 AND a.literal_kind = 'mapping' AND m.states_only_fields
          AND m.entry_count > 0
    """
    message = "`{qualified_name}` validates an explicit mapping with fields {fields}"
    examples = (
        Example(
            bad="""from pydantic import BaseModel

_NAME = "name"


class User(BaseModel):
    name: str


user = User.model_validate({"name": "Ada"})
other = User.model_validate({_NAME: "Grace"})
""",
            good="""from pydantic import BaseModel

_NAME = "name"


class User(BaseModel):
    name: str


user = User(name="Ada")
other = User.model_validate({_NAME: "Grace"})
""",
        ),
    )
    # The same validation through the constructor, the receiver kept as written.
    fix = """
        SELECT finding_key, path, start_line, start_column, end_line, end_column,
               CASE WHEN receiver <> '' THEN receiver || '(' || keywords || ')' END AS text
        FROM finding
    """

    constants = {"keywords": kwlist, "identifier": _IDENTIFIER}


class OptionalVariantDiscriminatedUnionCandidate(Rule, id="PY-PYDA0005"):
    """Optional-field models whose validator proves mutually exclusive variants."""

    minimum_variants: int = 2

    query = """
        SELECT f.*, c.value FROM pydantic_model_facts f JOIN (
          SELECT fact_id, count(*) AS value FROM pydantic_model_models_validators
          WHERE kind = 'model_after' AND proves_disjoint_optional_variants
            AND variant_count >= $minimum_variants
          GROUP BY fact_id) c ON c.fact_id = f.fact_id
    """
    message = "optional variant discriminated union candidate is {value} for `{fact_id}`"


class ConstructorModelCandidate(Rule, id="PY-PYDA0006"):
    """Constructor-heavy validated data classes that should be models."""

    minimum_parameters: int = 3
    minimum_attributes: int = 3
    minimum_validations: int = 1
    minimum_defaults: int = 1

    query = """
        SELECT f.*, c.value FROM pydantic_model_facts f JOIN (
          SELECT fact_id, count(*) AS value FROM pydantic_model_models
          WHERE is_undecorated_plain_class AND synchronous_init_count = 1
            AND fixed_parameter_count >= $minimum_parameters
            AND stored_parameter_count >= $minimum_attributes
            AND validation_count >= $minimum_validations AND default_count >= $minimum_defaults
            AND has_only_data_identity_methods
          GROUP BY fact_id) c ON c.fact_id = f.fact_id
    """
    message = "constructor model candidate is {value} for `{fact_id}`"
