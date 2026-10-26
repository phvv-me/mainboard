"""Rules over the fields and flexibility of Pydantic models (mcmr's PY-PYDA fields family)."""

from ....catalog import Example, Function, Rule

# A variadic tuple whose element type holds at most one level of brackets: nested ones are
# reached by applying the replacement again from the inside out.
_VARIADIC_TUPLE = r"(?:typing\.)?[Tt]uple\[((?:[^\[\]]|\[[^\[\]]*\])*), \.\.\.\]"


class PydanticSequenceAnnotation(Function):
    """An annotation with every variadic `tuple[T, ...]` written as `Sequence[T]`, however deep
    they nest."""

    annotation: str

    expression = (
        "regexp_replace(regexp_replace(regexp_replace(regexp_replace(annotation, "
        f"'{_VARIADIC_TUPLE}', 'Sequence[\\1]', 'g'), "
        f"'{_VARIADIC_TUPLE}', 'Sequence[\\1]', 'g'), "
        f"'{_VARIADIC_TUPLE}', 'Sequence[\\1]', 'g'), "
        f"'{_VARIADIC_TUPLE}', 'Sequence[\\1]', 'g')"
    )


class VariadicTupleModelField(Rule, id="PY-PYDA0007", applicability="review"):
    """Pydantic fields that require a homogeneous tuple representation."""

    query = """
        SELECT f.*, m.name AS model_name, f.span_path AS path, f.span_start_line AS start_line,
               f.span_start_column AS start_column, f.span_end_line AS end_line,
               f.span_end_column AS end_column
        FROM pydantic_model_models_fields f
        JOIN pydantic_model_models m ON m.fact_id = f.fact_id AND m.record_id = f.parent_id
        WHERE m.is_pydantic_model AND f.contains_variadic_tuple
    """
    message = (
        "`{model_name}.{name}` uses `{annotation}`, which fixes a homogeneous collection to "
        "tuple identity"
    )
    examples = (
        Example(
            bad="""from pydantic import BaseModel


class Ranking(BaseModel):
    scores: tuple[float, ...]
""",
            good="""from pydantic import BaseModel
from collections.abc import Sequence


class Ranking(BaseModel):
    scores: Sequence[float]
""",
        ),
    )
    # The annotation is the finding's span: every variadic tuple in it becomes `Sequence`
    # (a frozen model that is handed a list now keeps the list, which is unhashable). The
    # file's own `Sequence` is reused when it is the standard one, else the finding is left.
    fix = """
        WITH rewritten AS (
          SELECT f.*, pydantic_sequence_annotation(f.annotation) AS sequence_annotation,
                 (SELECT any_value(i.module) FROM import_binding_facts i
                  WHERE i.path = f.path AND i.name = 'Sequence') AS bound_module
          FROM finding f
        ), fixable AS (
          SELECT *, sequence_annotation <> annotation
                    AND coalesce(bound_module IN ('collections.abc', 'typing'), true) AS usable
          FROM rewritten
        )
        SELECT finding_key, path, start_line, start_column, end_line, end_column,
               CASE WHEN usable THEN sequence_annotation END AS text,
               CASE WHEN usable AND bound_module IS NULL
                    THEN 'from collections.abc import Sequence' END AS "import"
        FROM fixable
    """


class ImplicitArbitraryTypeModel(Rule, id="PY-PYDA0008"):
    """Pydantic models that permit arbitrary field types globally."""

    query = """
        SELECT *, flexible_base_span_path AS path, flexible_base_span_start_line AS start_line,
               flexible_base_span_start_column AS start_column,
               flexible_base_span_end_line AS end_line, flexible_base_span_end_column AS end_column
        FROM pydantic_model_models WHERE uses_flexible_model
    """
    message = "`{name}` permits arbitrary field types through `FrozenFlexModel`"
