"""Rules over how Pydantic models validate (mcmr's PY-PYDA validation family)."""

from ....catalog import Rule


class SingleFieldModelValidator(Rule, id="PY-PYDA0001"):
    """Model validators that depend on only one declared field."""

    query = """
        SELECT f.*, c.value FROM pydantic_model_facts f JOIN (
          SELECT v.fact_id, count(*) AS value FROM pydantic_model_models_validators v
          JOIN (SELECT parent_id, count(DISTINCT string_value) AS distinct_field_count
                FROM pydantic_model_values WHERE relation = 'models.validators.fields_read'
                GROUP BY parent_id) r ON r.parent_id = v.record_id
          WHERE v.kind = 'model_after' AND v.fields_read_present AND v.fields_read_length > 0
            AND r.distinct_field_count = 1 AND NOT v.has_self_call AND NOT v.has_nonfield_access
          GROUP BY v.fact_id) c ON c.fact_id = f.fact_id
    """
    message = "single field model validator is {value} for `{fact_id}`"


class DeclarativeFieldConstraintCandidate(Rule, id="PY-PYDA0002"):
    """Field validators that reimplement built-in Pydantic constraints."""

    query = """
        SELECT f.*, c.value FROM pydantic_model_facts f JOIN (
          SELECT fact_id, sum(declarative_constraint_count) AS value
          FROM pydantic_model_models_validators WHERE kind = 'field' GROUP BY fact_id
        ) c ON c.fact_id = f.fact_id WHERE c.value > 0
    """
    message = "declarative field constraint candidate is {value} for `{fact_id}`"


class ImperativeModelInputValidation(Rule, id="PY-PYDA0003"):
    """Model factories that manually reproduce Pydantic field validation."""

    query = """
        SELECT * FROM function_functions
        WHERE is_model_method AND NOT is_pydantic_validator AND checks_raw_input_type
          AND raises_validation_exception AND constructs_owner_model
    """
    message = "`{fact_id}` satisfies imperative model input validation"
