"""The stages of the syntax relations that rank the operands of a node, place it in its parent and
keep the nodes a rule can ask about (see `core.py` for the whole chain)."""

from .kinds import fill

_RANKS = r"""
    SELECT file, id, rk, nxt FROM (
      SELECT file, id,
             count(*) FILTER (WHERE opnd) OVER (PARTITION BY file, pid ORDER BY id) AS rk,
             lead(type) OVER (PARTITION BY file ORDER BY id) AS nxt
      FROM syntax_tree)
"""

_PLACES = r"""
    WITH placed AS (
      SELECT n.*, p.type AS ptype, g.type AS gtype, r.rk, r.nxt
      FROM syntax_tree n
      JOIN syntax_ranks r ON r.file = n.file AND r.id = n.id
      LEFT JOIN syntax_tree p ON p.file = n.file AND p.id = n.pid
      LEFT JOIN syntax_tree g ON g.file = p.file AND g.id = p.pid
      WHERE n.type IN @ALIVE@ OR n.dc > 0 OR n.type IN ('and', 'or')
    ), delegations AS (
      SELECT file, pid FROM syntax_tree WHERE type = 'from' AND sib = 1
    )
    SELECT placed.* EXCLUDE (gtype), CASE
        WHEN type NOT IN @ALIVE@ THEN false
        WHEN ptype IS NULL THEN true
        WHEN ptype = 'yield' THEN yf.pid IS NULL
        WHEN ptype = 'attribute' THEN rk = 1
        WHEN ptype IN ('for_in_clause', 'keyword_argument', 'named_expression',
                       'default_parameter') THEN rk = 2
        WHEN ptype = 'typed_default_parameter' THEN type = 'type' OR rk = 2
        WHEN ptype = 'typed_parameter' THEN type = 'type'
        WHEN ptype IN ('assignment', 'augmented_assignment') THEN type = 'type' OR rk >= 2
        WHEN ptype IN ('raise_statement', 'assert_statement', 'interpolation', 'lambda',
                       'decorator', 'as_pattern', 'with_item') THEN rk = 1
        WHEN ptype IN ('except_clause', 'except_group_clause', 'case_clause', 'match_statement',
                       'elif_clause') THEN type = 'block'
        WHEN ptype = 'function_definition'
          THEN type IN ('type_parameter', 'parameters', 'type', 'block')
        WHEN ptype = 'class_definition' THEN type IN ('type_parameter', 'argument_list', 'block')
        WHEN ptype = 'constrained_type' THEN sib = 1
        WHEN ptype = 'type_parameter'
          THEN gtype NOT IN ('function_definition', 'class_definition') OR nxt = 'constrained_type'
        WHEN ptype = 'argument_list' AND gtype = 'class_definition'
          THEN type NOT IN ('keyword_argument', 'dictionary_splat')
        WHEN ptype = 'parameters'
          THEN type IN ('default_parameter', 'typed_parameter', 'typed_default_parameter')
        WHEN ptype IN ('slice', 'global_statement', 'nonlocal_statement', 'import_statement',
                       'import_from_statement', 'future_import_statement', 'print_statement',
                       'exec_statement', 'type_alias_statement') THEN false
        WHEN ptype IN ('string', 'concatenated_string')
          THEN type = 'interpolation' OR (ptype = 'concatenated_string' AND type = 'string')
        ELSE true END AS ok
    FROM placed LEFT JOIN delegations yf ON yf.file = placed.file AND yf.pid = placed.pid
                                        AND placed.ptype = 'yield'
"""

_KEPT = r"""
    WITH live AS (
      SELECT * EXCLUDE (blocked, ok) FROM (
        SELECT *, max(CASE WHEN ok THEN -1 ELSE id + dc END)
                    OVER (PARTITION BY file ORDER BY id) AS blocked
        FROM syntax_places) WHERE ok AND blocked < id
    ), operators AS (
      SELECT file, pid AS id, type AS op FROM syntax_places
      WHERE ptype = 'boolean_operator' AND type IN ('and', 'or')
    )
    SELECT l.file, l.path, l.id, l.pid, l.type, l.dc, l.cc, l.sl, l.sc, l.el, l.ec, l.name, l.up,
           l.ptype, v.kind AS vkind,
           l.type IN ('function_definition', 'class_definition') AS is_def,
           CASE
             WHEN v.kind IS NULL THEN false
             WHEN l.type = 'expression_statement'
               THEN l.nxt NOT IN ('assignment', 'augmented_assignment')
             WHEN l.type IN ('assignment', 'augmented_assignment')
               THEN l.ptype = 'expression_statement'
             WHEN l.type = 'string' THEN l.ptype <> 'concatenated_string'
             WHEN l.type = 'boolean_operator'
               THEN NOT (l.ptype = 'boolean_operator' AND l.sib = 0 AND o.op = po.op)
             WHEN l.type IN ('lambda', 'await', 'yield') THEN l.dc > 0
             WHEN l.type IN ('expression_list', 'pattern_list')
               THEN l.ptype <> 'delete_statement'
             ELSE true END AS kept
    FROM live l
    LEFT JOIN (VALUES @VOCABULARY@) v(type, kind) ON v.type = l.type
    LEFT JOIN operators o ON o.file = l.file AND o.id = l.id
    LEFT JOIN operators po ON po.file = l.file AND po.id = l.pid
"""


def shapes() -> dict[str, str]:
    """The relations `ranks`, `places` and `kept`, each reading the stage before it by view name."""
    return {"ranks": _RANKS, "places": fill(_PLACES), "kept": fill(_KEPT)}
