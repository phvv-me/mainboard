"""Rules over docstrings (mcmr's PY-DOCU family)."""

from ....catalog import Rule

_LABELED_LINE = (
    r"(?m)^(?:(?i:args:|arguments:|returns:|parameters|returns)$|:param|:return|:rtype"
    r"|[A-Za-z_][A-Za-z0-9_]*(?: [A-Za-z0-9_]+){0,2}:\s*$)"
)
_SPACE = " \t\n\r\x0b\x0c"


class CompactHouseDocstring(Rule, id="PY-DOCU0001"):
    """Docstrings whose summary is no finished sentence or whose body carries headings.

    A label is a line of at most three words ending in a colon, so prose that runs into a colon
    is not one. A cyclopts command keeps its `Args:` heading, since cyclopts reads the parameter
    help under it: a function decorated `*.command` or `*.default`, one handed to such a call by
    name (`group.command(verbs.submit)`), or a function named through its module or class
    (`lint.check`, `JobVerbs.show`) or listed bare in a tuple or list (`_COMMANDS = (sql, move)`)
    in a module that makes such calls, the shape of a route table.
    """

    maximum_summary: int = 99

    query = f"""
        WITH literals AS (
          SELECT path, start_line, start_column, name, docstring AS literal,
                 EXISTS (SELECT 1 FROM core_decor x
                         WHERE x.file_id = d.file_id AND x.def_id = d.def_id
                           AND regexp_extract(trim(split_part(x.name, '(', 1)), '[^.]*$')
                               IN ('command', 'default'))
                 OR EXISTS (SELECT 1 FROM call_calls c
                            JOIN call_expressions e ON e.call_id = c.call_id
                              AND e.root_relation = 'argument' AND e.root_ordinal = 0
                              AND e.depth = 0
                            WHERE regexp_extract(c.callee_text, '[^.]*$') IN ('command', 'default')
                              AND regexp_extract(e.text, '[^.]*$') = d.name)
                 OR EXISTS (SELECT 1 FROM ast_nodes a
                            JOIN ast_nodes o ON o.fact_id = a.fact_id AND o.parent_id = a.node_id
                              AND o.sibling_index = 0 AND o.type = 'identifier'
                            JOIN ast_nodes m ON m.fact_id = a.fact_id AND m.parent_id = a.node_id
                              AND m.sibling_index >= 1 AND m.type = 'identifier'
                            WHERE a.type = 'attribute' AND m.name = d.name
                              AND a.path IN (SELECT r.path FROM call_calls r
                                             WHERE regexp_extract(r.callee_text, '[^.]*$')
                                                   IN ('command', 'default')))
                 OR EXISTS (SELECT 1 FROM ast_nodes i
                            JOIN ast_nodes t ON t.fact_id = i.fact_id AND t.node_id = i.parent_id
                              AND t.type IN ('tuple', 'list')
                            WHERE i.type = 'identifier' AND i.name = d.name
                              AND i.path IN (SELECT r.path FROM call_calls r
                                             WHERE regexp_extract(r.callee_text, '[^.]*$')
                                                   IN ('command', 'default'))) AS is_command
          FROM core_def d WHERE type = 'function_definition' AND language = 'python'
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
        FROM split
        WHERE regexp_matches(if(is_command, regexp_replace(body, '(?im)^args:$', '', 'g'), body),
                             $labeled)
    """

    constants = {"labeled": _LABELED_LINE}


class TensorDocstringSemantics(Rule, id="PY-DOCU0002"):
    """Public tensor callables whose docstring states no shape or no dtype semantics."""

    query = """
            SELECT * FROM function_functions f
            WHERE visibility = 'public'
              AND NOT (has_tensor_shape_semantics AND has_tensor_dtype_semantics)
              AND EXISTS (SELECT 1 FROM function_tensor_roles r WHERE r.function_id = f.entity_id)
        """
    message = "`{fact_id}` satisfies tensor docstring semantics"
