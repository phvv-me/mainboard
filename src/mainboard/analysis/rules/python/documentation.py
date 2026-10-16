"""Rules over docstrings (mcmr's PY-DOCU family)."""

from ...catalog import Rule
from ...facts import Setting

_LABELED_LINE = (
    r"(?m)^(?:(?i:args:|arguments:|returns:|parameters|returns)$|:param|:return|:rtype"
    r"|[A-Za-z_][A-Za-z0-9_ ]*:\s*$)"
)
_SPACE = " \t\n\r\x0b\x0c"


class CompactHouseDocstring(Rule, id="PY-DOCU0001"):
    """Docstrings whose summary is no finished sentence or whose body carries headings."""

    maximum_summary: int = 99

    query = f"""
        WITH literals AS (
          SELECT path, start_line, start_column, name, docstring AS literal
          FROM core_def WHERE type = 'function_definition' AND language = 'python'
            AND docstring IS NOT NULL
        ), spelled AS (
          SELECT *, regexp_replace(literal, '^[uUrR]*', '') AS bare,
                 regexp_matches(literal, '^[uU]?[rR]') AS is_raw
          FROM literals
        ), decoded AS (
          SELECT *, functions_docstring(CASE WHEN is_raw THEN content
                                             ELSE functions_unescaped(content) END) AS docstring
          FROM (SELECT *, CASE WHEN regexp_matches(bare, '^(["]{{3}}|[\\x27]{{3}})')
                               THEN substr(bare, 4, length(bare) - 6)
                               ELSE substr(bare, 2, length(bare) - 2) END AS content
                FROM spelled)
        ), documented AS (
          SELECT *, trim(docstring, '{_SPACE}') AS text FROM decoded WHERE docstring <> ''
        ), split AS (
          SELECT *, trim(str_split(text, chr(10))[1], '{_SPACE}') AS summary,
                 regexp_replace(text, '(?s)^[^\\n]*\\n?', '', 'g') AS body
          FROM documented
        )
        SELECT path, start_line, start_column,
          'the docstring of `' || name || '` opens with ' || length(summary)
            || ' characters that do not read as one finished sentence' AS message
        FROM split
        WHERE summary = '' OR length(summary) > $maximum_summary
           OR right(summary, 1) NOT IN ('.', '!', '?')
           OR starts_with(lower(summary), 'see ') OR starts_with(lower(summary), 'refer to ')
        UNION ALL
        SELECT path, start_line, start_column, 'the docstring of `' || name
          || '` carries a heading or a label where this project writes plain lines'
        FROM split WHERE regexp_matches(body, $labeled)
    """

    def parameters(self) -> dict[str, Setting]:
        labeled = _LABELED_LINE
        return super().parameters() | {"labeled": labeled}


class TensorDocstringSemantics(Rule, id="PY-DOCU0002"):
    """Public tensor callables whose docstring states no shape or no dtype semantics."""

    query = """
            SELECT * FROM function_functions f
            WHERE visibility = 'public'
              AND NOT (has_tensor_shape_semantics AND has_tensor_dtype_semantics)
              AND EXISTS (SELECT 1 FROM function_tensor_roles r WHERE r.function_id = f.entity_id)
        """
    message = "`{fact_id}` satisfies tensor docstring semantics"
