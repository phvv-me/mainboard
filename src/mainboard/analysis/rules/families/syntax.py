"""mcmr's SyntaxFact, derived from sitting_duck's Python syntax trees.

mcmr keeps one fact per function or class, holding a language-neutral tree of what its ruff AST
states: the statements and expressions a rule asks about (a call, a binding, a loop, a name), none
of the grammar around them. Tree-sitter's Python grammar is finer: it has a node for the block, the
argument list, the keyword's name, the target of every assignment. The derivation walks the
tree-sitter tree once and keeps what ruff's AST would have:

- a node is kept when its type names a statement or an expression and the place it sits in states
  it (the left side of an assignment, the attribute after a dot and the name of a keyword argument
  are not expressions the program evaluates); everything below a place that does not state its
  children is dropped by one running maximum over the preorder;
- wrappers that carry no meaning (blocks, argument lists, parentheses, clauses) are passed through,
  so a node's parent is its nearest kept ancestor;
- a nested function or class stays a leaf of the fact that holds it and is the root of its own;
- ordinals number the kept nodes of one fact in source order, depth counts the kept ancestors;
- a tuple the grammar leaves implicit (`a(), b()` as a statement, `x[i, j]`, `A[B, C]`) gets a
  synthetic collection node, as ruff's Tuple;
- a file with a tree-sitter error holds no fact, as when ruff cannot parse it.

Relations rules read: `facts` (one declaration each), `nodes` (the kept tree, preorder) and
`children` (parent-child links in order). The rest are the stages of `nodes`, each stored in the
fact base so no query holds more than one stage's joins (one query for the whole tree needed
more than 3 GB on research/reproducibility): `lines` (offset of every line), `tree` (the Python
nodes with the columns the stages read), `ranks`/`places` (a node's operand rank and the place
it sits in), `kept` (the nodes that survive), `owners`, `decorators`, `owned`, `bare`, `tuples`,
`ends`, `members` (what each fact holds, with the synthetic and boundary nodes), `links`/`climb`
(each member's nearest kept ancestor), `swaps` and `depths` (depth and operand order),
`numbered`/`spanned`/`parented` (ordinals, subtree ends, parents), `targets`/`dotted` (the names
of assignments and calls) and `frames` (each fact's byte offset). Each stage reads the earlier
ones by view name.
"""

from ...catalog import Family

_STATEMENTS = {
    "function_definition": "callable",
    "class_definition": "type",
    "return_statement": "return",
    "raise_statement": "raise",
    "if_statement": "branch",
    "for_statement": "loop",
    "while_statement": "loop",
    "try_statement": "guard",
    "with_statement": "scope",
    "import_statement": "import",
    "import_from_statement": "import",
    "future_import_statement": "import",
    "assignment": "binding",
    "augmented_assignment": "binding",
    "expression_statement": "effect",
    "pass_statement": "statement",
    "break_statement": "statement",
    "continue_statement": "statement",
    "global_statement": "statement",
    "nonlocal_statement": "statement",
    "assert_statement": "statement",
    "delete_statement": "statement",
    "match_statement": "statement",
    "type_alias_statement": "statement",
    "print_statement": "statement",
    "exec_statement": "statement",
}

_EXPRESSIONS = {
    "identifier": "name",
    "attribute": "member",
    "call": "call",
    "string": "text",
    "concatenated_string": "text",
    "integer": "literal",
    "float": "literal",
    "true": "literal",
    "false": "literal",
    "none": "literal",
    "ellipsis": "expression",
    "list": "collection",
    "set": "collection",
    "tuple": "collection",
    "dictionary": "collection",
    "expression_list": "collection",
    "pattern_list": "collection",
    "tuple_pattern": "collection",
    "list_pattern": "collection",
    "list_comprehension": "comprehension",
    "set_comprehension": "comprehension",
    "dictionary_comprehension": "comprehension",
    "generator_expression": "comprehension",
    "lambda": "callable",
    "binary_operator": "operation",
    "boolean_operator": "operation",
    "comparison_operator": "operation",
    "unary_operator": "operation",
    "not_operator": "operation",
    "union_type": "operation",
    "subscript": "index",
    "generic_type": "index",
    "await": "await",
    "conditional_expression": "expression",
    "yield": "expression",
    "list_splat": "expression",
    "list_splat_pattern": "expression",
    "named_expression": "expression",
    "slice": "expression",
}

# Nodes that carry no meaning of their own: their children belong to the nearest kept ancestor.
_TRANSPARENT = (
    "module",
    "block",
    "parenthesized_expression",
    "argument_list",
    "pair",
    "dictionary_splat",
    "keyword_argument",
    "as_pattern",
    "with_item",
    "with_clause",
    "decorated_definition",
    "decorator",
    "elif_clause",
    "else_clause",
    "except_clause",
    "except_group_clause",
    "finally_clause",
    "case_clause",
    "for_in_clause",
    "if_clause",
    "interpolation",
    "type",
    "type_parameter",
    "constrained_type",
    "parameters",
    "default_parameter",
    "typed_parameter",
    "typed_default_parameter",
)

# What can stand as an operand, to number the operands of a node: `a.b` has the object first.
_OPERANDS = (
    *_EXPRESSIONS,
    "parenthesized_expression",
    "pair",
    "dictionary_splat",
    "keyword_argument",
    "as_pattern",
    "with_item",
    "assignment",
)

# What tree-sitter nests around a star it misreads: `[*a.b()]` is `call(attribute(list_splat(a)))`.
_CHAINED = (
    "attribute",
    "call",
    "subscript",
    "binary_operator",
    "boolean_operator",
    "comparison_operator",
    "conditional_expression",
)

# Statements that end where their last simple statement ends: tree-sitter lets a trailing comment
# at the end of a block stretch every compound statement around it.
_SIMPLE = (
    "expression_statement",
    "return_statement",
    "raise_statement",
    "pass_statement",
    "break_statement",
    "continue_statement",
    "global_statement",
    "nonlocal_statement",
    "assert_statement",
    "delete_statement",
    "import_statement",
    "import_from_statement",
    "future_import_statement",
    "print_statement",
    "exec_statement",
    "type_alias_statement",
)

_COMPOUND = (
    "function_definition",
    "class_definition",
    "if_statement",
    "for_statement",
    "while_statement",
    "try_statement",
    "with_statement",
    "match_statement",
)


def _list(names: tuple[str, ...] | list[str]) -> str:
    """A SQL list of string literals."""
    return "(" + ", ".join(f"'{name}'" for name in names) + ")"


def _values(vocabulary: dict[str, str]) -> str:
    return ", ".join(f"('{type}', '{kind}')" for type, kind in vocabulary.items())


_LINES = """
    SELECT path, line, text, sum(strlen(text) + 1) OVER (PARTITION BY path ORDER BY line)
                               - strlen(text) - 1 AS off
    FROM (SELECT path, generate_subscripts(parts, 1) AS line, unnest(parts) AS text
          FROM (SELECT path, string_split(content, chr(10)) AS parts FROM source_files
                WHERE path IN (SELECT DISTINCT path FROM ast_files WHERE language = 'python')))
"""


def _fill(sql: str) -> str:
    """The placeholders of a SQL template, replaced by the vocabulary they name."""
    return (
        sql.replace("@VOCABULARY@", _values({**_STATEMENTS, **_EXPRESSIONS}))
        .replace("@ALIVE@", _list([*_STATEMENTS, *_EXPRESSIONS, *_TRANSPARENT]))
        .replace("@OPERANDS@", _list(_OPERANDS))
        .replace("@CHAINED@", _list(_CHAINED))
        .replace("@FACT_TYPES@", _list(["decorated_definition", *_COMPOUND[:2], *_SIMPLE]))
        .replace("@SIMPLE@", _list(_SIMPLE))
        .replace("@COMPOUND@", _list(_COMPOUND))
    )


_FACTS = r"""
    WITH python AS (
      SELECT a.fact_id AS file, a.path, a.node_id AS id, a.parent_id AS pid, a.type,
             a.descendant_count AS dc, a.start_line AS sl, a.start_column AS sc,
             a.end_line AS el, a.end_column AS ec, a.name, a.qualified_name
      FROM ast_nodes a JOIN ast_files f ON f.fact_id = a.fact_id
      WHERE f.language = 'python' AND a.type IN @FACT_TYPES@
        AND a.fact_id NOT IN (SELECT fact_id FROM ast_nodes WHERE type = 'ERROR')
    ), simples AS (
      SELECT * FROM python WHERE type IN @SIMPLE@
    ), definitions AS (
      SELECT d.file, d.path, d.id, d.type, d.id + d.dc AS last, d.qualified_name,
             coalesce(p.sl, d.sl) AS sl, coalesce(p.sc, d.sc) AS sc
      FROM python d LEFT JOIN python p
        ON p.file = d.file AND p.id = d.pid AND p.type = 'decorated_definition'
      WHERE d.type IN ('function_definition', 'class_definition')
    ), ended AS (
      SELECT d.*, s.el, s.ec
      FROM definitions d ASOF JOIN simples s ON s.file = d.file AND d.last >= s.id
    )
    SELECT row_number() OVER (ORDER BY e.path, e.id) - 1 AS fact_order,
           'syntax:' || e.path || ':' || e.sl || ':' || e.sc || ':' || e.qualname AS fact_id,
           e.path, e.sl AS start_line, e.sc AS start_column, e.el AS end_line, e.ec AS end_column,
           'python' AS language, e.qualname,
           CASE e.type WHEN 'function_definition' THEN 'callable' ELSE 'type' END AS kind,
           e.file AS ast_file, e.id AS ast_node,
           syntax_slice(s.content, e.sl, e.sc, e.el, e.ec) AS source
    FROM (SELECT *, array_to_string(list_transform(qualified_name, lambda x: x.name), '.')
                      AS qualname FROM ended) e
    JOIN source_files s ON s.path = e.path
"""

_TREE = r"""
    WITH splats AS (
      SELECT fact_id AS file, node_id AS tail FROM ast_nodes
      WHERE type = 'list_splat' AND node_id = parent_id + 1
    ), above AS (
      SELECT file, tail, id, tail - id AS k, type, dc, cc, el, ec FROM (
        SELECT s.file, s.tail, a.node_id AS id, a.type, a.descendant_count AS dc,
               a.children_count AS cc, a.end_line AS el, a.end_column AS ec,
               row_number() OVER (PARTITION BY s.file, s.tail ORDER BY a.node_id DESC) AS rn
        FROM splats s
        JOIN ast_nodes a ON a.fact_id = s.file AND a.node_id < s.tail
                        AND a.node_id + a.descendant_count >= s.tail
        WHERE a.type IN @CHAINED@) WHERE tail - id = rn
    ), moved AS (
      SELECT c.file, c.id + 1 AS id, c.k, c.type, c.dc - 1 AS dc, c.cc, x.start_line AS sl,
             x.start_column AS sc, c.el, c.ec
      FROM above c JOIN ast_nodes x ON x.fact_id = c.file AND x.parent_id = c.tail
                                   AND x.sibling_index = 1
    ), tops AS (
      SELECT file, id FROM above QUALIFY k = max(k) OVER (PARTITION BY file, tail)
    )
    SELECT a.fact_id AS file, a.path, a.node_id AS id,
           a.parent_id + CASE WHEN h.id IS NOT NULL AND a.node_id <> a.parent_id + 1 THEN 1
                              ELSE 0 END AS pid,
           CASE WHEN m.id IS NOT NULL THEN m.type WHEN t.id IS NOT NULL THEN 'list_splat'
                ELSE a.type END AS type,
           a.sibling_index AS sib, coalesce(m.dc, a.descendant_count) AS dc,
           coalesce(m.cc, a.children_count) AS cc, coalesce(m.sl, a.start_line) AS sl,
           coalesce(m.sc, a.start_column) AS sc, coalesce(m.el, a.end_line) AS el,
           coalesce(m.ec, a.end_column) AS ec,
           CASE WHEN a.type IN ('identifier', 'function_definition', 'class_definition')
                THEN a.name WHEN a.type = 'string' THEN substr(a.name, 1, 4) ELSE '' END AS name,
           greatest(a.scope.function, a.scope.class) AS up,
           a.type IN @OPERANDS@ AND NOT (a.type IN ('lambda', 'await', 'yield')
                                         AND a.descendant_count = 0) AS opnd,
           CASE WHEN m.k = 1 THEN 1 ELSE 0 END AS lead
    FROM ast_nodes a JOIN ast_files f ON f.fact_id = a.fact_id
    LEFT JOIN above h ON h.file = a.fact_id AND h.id = a.parent_id
    LEFT JOIN moved m ON m.file = a.fact_id AND m.id = a.node_id
    LEFT JOIN tops t ON t.file = a.fact_id AND t.id = a.node_id
    WHERE f.language = 'python'
      AND a.fact_id NOT IN (SELECT fact_id FROM ast_nodes WHERE type = 'ERROR')
"""

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

_OWNERS = r"""
    WITH RECURSIVE r(file, id, owner) AS (
      SELECT file, id, id FROM syntax_kept WHERE is_def
      UNION ALL
      SELECT s.file, s.id, r.owner FROM r
      JOIN (SELECT file, id, up FROM syntax_tree WHERE type = 'lambda' AND dc > 0) s
        ON s.file = r.file AND s.up = r.id
    ) SELECT * FROM r
"""

_DECORATORS = r"""
    SELECT d.file, d.id, d.id + d.dc AS last, k.id AS owner
    FROM syntax_kept d JOIN syntax_kept k ON k.file = d.file AND k.pid = d.pid AND k.is_def
    WHERE d.type = 'decorator'
"""

_OWNED = r"""
    SELECT k.file, k.path, k.id, k.dc, k.type, k.name, k.sl, k.sc, k.el, k.ec, k.vkind,
           coalesce(d.last >= k.id, false) AS decorating,
           CASE WHEN d.last >= k.id THEN d.owner ELSE ow.owner END AS owner
    FROM (SELECT * FROM syntax_kept WHERE kept AND NOT is_def) k
    ASOF LEFT JOIN syntax_decorators d ON d.file = k.file AND k.id >= d.id
    LEFT JOIN syntax_owners ow ON ow.file = k.file AND ow.id = k.up
"""

_BARE = r"""
    SELECT g.file, g.id, a.sl, a.sc, z.el, z.ec
    FROM syntax_kept g
    JOIN syntax_tree a ON a.file = g.file AND a.pid = g.id AND a.sib = 1
    JOIN syntax_tree z ON z.file = g.file AND z.pid = g.id AND z.sib = g.cc - 2
    WHERE g.type = 'generator_expression' AND g.ptype = 'call'
"""

_TUPLES = r"""
    SELECT a.file, a.subscript, a.key, a.first_id, a.last_eid, a.sl, a.sc, a.el,
           a.ec + CASE WHEN decode(array_slice(encode(l.text), a.ec + 1, a.ec + 1)) = ','
                       THEN 1 ELSE 0 END AS ec
    FROM (
      SELECT s.file, s.path, s.id AS subscript, -s.id - 1000000000 AS key, min(g.id) AS first_id,
             max(g.id + g.dc) AS last_eid, arg_min(g.sl, g.id) AS sl, arg_min(g.sc, g.id) AS sc,
             arg_max(g.el, g.id) AS el, arg_max(g.ec, g.id) AS ec
      FROM syntax_kept s
      JOIN syntax_places g ON g.file = s.file AND g.pid = s.id AND g.opnd AND g.rk >= 2
      WHERE s.type = 'subscript' AND s.kept
      GROUP BY s.file, s.path, s.id HAVING count(*) >= 2
      UNION ALL
      SELECT g.file, g.path, g.id, -g.id - 1000000000, min(t.id), max(t.id + t.dc),
             arg_min(t.sl, t.id), arg_min(t.sc, t.id), arg_max(t.el, t.id), arg_max(t.ec, t.id)
      FROM syntax_kept g
      JOIN syntax_places p ON p.file = g.file AND p.pid = g.id AND p.type = 'type_parameter'
      JOIN syntax_places t ON t.file = p.file AND t.pid = p.id AND t.type = 'type'
      WHERE g.type = 'generic_type' AND g.kept
      GROUP BY g.file, g.path, g.id HAVING count(*) >= 2
      UNION ALL
      SELECT s.file, s.path, s.id, -s.id - 1000000000, min(g.id), max(g.id + g.dc),
             arg_min(g.sl, g.id), arg_min(g.sc, g.id), arg_max(g.el, g.id), arg_max(g.ec, g.id)
      FROM syntax_kept s
      JOIN syntax_places g ON g.file = s.file AND g.pid = s.id AND g.opnd
      WHERE s.type = 'expression_statement' AND s.kept
      GROUP BY s.file, s.path, s.id
      HAVING count(*) >= 2 OR bool_or(s.el <> g.el OR s.ec <> g.ec)
    ) a LEFT JOIN syntax_lines l ON l.path = a.path AND l.line = a.el
"""

_ENDS = r"""
    WITH simples AS (
      SELECT file, id, el, ec FROM syntax_tree WHERE type IN @SIMPLE@
    ), compounds AS (
      SELECT file, id, id + dc AS last FROM syntax_kept
      WHERE type IN @COMPOUND@ AND (kept OR is_def)
    )
    SELECT c.file, c.id, s.el, s.ec FROM compounds c
    ASOF JOIN simples s ON s.file = c.file AND c.last >= s.id
"""

_MEMBERS = r"""
    SELECT o.file, o.path, o.id AS key, o.id, o.id AS sid, o.id + o.dc AS eid, o.owner,
           o.decorating, false AS leaf, false AS root, o.type, o.name,
           coalesce(b.sl, o.sl) AS sl, coalesce(b.sc, o.sc) AS sc,
           coalesce(e.el, b.el, o.el) AS el, coalesce(e.ec, b.ec, o.ec) AS ec,
           CASE WHEN o.type = 'string' AND regexp_matches(lower(o.name), '^(b|br|rb)[''"]')
                THEN 'expression' ELSE o.vkind END AS kind, false AS synthetic
    FROM syntax_owned o LEFT JOIN syntax_ends e ON e.file = o.file AND e.id = o.id
    LEFT JOIN syntax_bare b ON b.file = o.file AND b.id = o.id
    WHERE o.owner IS NOT NULL
    UNION ALL
    SELECT t.file, o.path, t.key, t.first_id, t.first_id, t.last_eid, o.owner, o.decorating,
           false, false, 'subscript_tuple', '', t.sl, t.sc, t.el, t.ec, 'collection', true
    FROM syntax_tuples t JOIN syntax_owned o ON o.file = t.file AND o.id = t.subscript
    UNION ALL
    SELECT d.file, d.path, d.id, d.id, -1, d.id + d.dc, d.id, false, false, true, d.type, d.name,
           coalesce(p.sl, d.sl), coalesce(p.sc, d.sc), e.el, e.ec, d.vkind, false
    FROM syntax_kept d LEFT JOIN syntax_tree p ON p.file = d.file AND p.id = d.pid
                                              AND p.type = 'decorated_definition'
    JOIN syntax_ends e ON e.file = d.file AND e.id = d.id
    WHERE d.is_def
    UNION ALL
    SELECT d.file, d.path, -d.id - 1, d.id, coalesce(p.id, d.id), d.id + d.dc, ow.owner, false,
           true, false, d.type, d.name, coalesce(p.sl, d.sl), coalesce(p.sc, d.sc), e.el, e.ec,
           d.vkind, false
    FROM syntax_kept d LEFT JOIN syntax_tree p ON p.file = d.file AND p.id = d.pid
                                              AND p.type = 'decorated_definition'
    JOIN syntax_ends e ON e.file = d.file AND e.id = d.id
    JOIN syntax_owners ow ON ow.file = d.file AND ow.id = d.up
    WHERE d.is_def
"""

_LINKS = r"""
    SELECT t.file, t.id, CASE WHEN t.type = 'decorator' THEN dc.owner ELSE t.pid END AS above,
           t.kept OR t.is_def AS holds
    FROM syntax_kept t LEFT JOIN syntax_decorators dc ON dc.file = t.file AND dc.id = t.id
"""

_CLIMB = r"""
    WITH RECURSIVE c(file, key, cand) AS (
      SELECT m.file, m.key, l.above FROM syntax_members m
      JOIN syntax_links l ON l.file = m.file AND l.id = m.id WHERE NOT m.root
      UNION ALL
      SELECT c.file, c.key, l.above FROM c
      JOIN syntax_links l ON l.file = c.file AND l.id = c.cand WHERE NOT l.holds
    )
    SELECT c.file, c.key, c.cand AS parent FROM c
    JOIN syntax_links l ON l.file = c.file AND l.id = c.cand WHERE l.holds
"""

_SWAPS = r"""
    SELECT f.file, f.id AS f1, f.id + f.dc AS f2, g.id AS s1, g.id + g.dc AS s2
    FROM syntax_places f
    JOIN syntax_places g ON g.file = f.file AND g.pid = f.pid AND g.rk = 2 AND g.opnd
    JOIN syntax_places p ON p.file = f.file AND p.id = f.pid
    WHERE f.rk = 1 AND f.opnd
      AND (p.type = 'conditional_expression'
           OR (p.type = 'pair' AND p.ptype = 'dictionary_comprehension'))
"""

_DEPTHS = r"""
    SELECT file, id, above, moved FROM (
      SELECT file, id, sum(opening) OVER w AS above, sum(shifting) OVER w AS moved
      FROM (
        SELECT file, id + 1 AS pos, 1 AS opening, 0 AS shifting, 0 AS probe, NULL::BIGINT AS id
        FROM syntax_kept WHERE kept OR is_def
        UNION ALL
        SELECT file, id + dc + 1, -1, 0, 0, NULL FROM syntax_kept WHERE kept OR is_def
        UNION ALL SELECT file, first_id, 1, 0, 0, NULL FROM syntax_tuples
        UNION ALL SELECT file, last_eid + 1, -1, 0, 0, NULL FROM syntax_tuples
        UNION ALL SELECT file, f1, 0, s2 - f2, 0, NULL FROM syntax_swaps
        UNION ALL SELECT file, f2 + 1, 0, f2 - s2, 0, NULL FROM syntax_swaps
        UNION ALL SELECT file, s1, 0, f1 - s1, 0, NULL FROM syntax_swaps
        UNION ALL SELECT file, s2 + 1, 0, s1 - f1, 0, NULL FROM syntax_swaps
        UNION ALL SELECT file, id, 0, 0, 1, id FROM (SELECT DISTINCT file, id FROM syntax_members))
      WINDOW w AS (PARTITION BY file ORDER BY pos, probe ROWS UNBOUNDED PRECEDING)
    ) WHERE id IS NOT NULL
"""

_NUMBERED = r"""
    SELECT *, row_number() OVER (PARTITION BY file, owner ORDER BY slot) - 1 AS ordinal
    FROM (SELECT m.*, CASE WHEN m.root THEN -1
                           ELSE 2 * (m.sid + d.moved) - CASE WHEN m.synthetic THEN 1 ELSE 0 END
                      END AS slot,
                 2 * (m.eid + d.moved) AS limit_slot
          FROM syntax_members m JOIN syntax_depths d ON d.file = m.file AND d.id = m.id)
"""

_SPANNED = r"""
    SELECT n.*, q.ordinal + 1 AS subtree_end
    FROM syntax_numbered n
    ASOF JOIN syntax_numbered q ON q.file = n.file AND q.owner = n.owner
                               AND n.limit_slot >= q.slot
"""

_PARENTED = r"""
    SELECT n.*, coalesce(x.key, c.parent) AS parent_key
    FROM syntax_spanned n
    LEFT JOIN syntax_climb c ON c.file = n.file AND c.key = n.key
    LEFT JOIN syntax_tuples x ON x.file = c.file AND x.subscript = c.parent AND n.id >= x.first_id
                             AND NOT n.synthetic
"""

_TARGETS = r"""
    SELECT p.file, p.pid AS id,
           CASE p.type WHEN 'identifier' THEN p.name WHEN 'attribute' THEN i.name ELSE '' END
             AS name
    FROM syntax_places p
    LEFT JOIN (SELECT file, id, name FROM syntax_tree WHERE type = 'identifier') i
           ON i.file = p.file AND i.id = p.id + p.dc
    WHERE p.ptype IN ('assignment', 'augmented_assignment', 'for_statement') AND p.rk = 1
      AND p.opnd
"""

_DOTTED = r"""
    WITH RECURSIVE identifiers AS (
      SELECT file, id, name FROM syntax_tree WHERE type = 'identifier'
    ), chains AS (
      SELECT file, id, type, dc,
             id + CASE type WHEN 'parenthesized_expression' THEN 2 ELSE 1 END + lead AS first
      FROM syntax_tree WHERE type IN ('attribute', 'call', 'parenthesized_expression') AND dc > 0
    ), q(file, id, val) AS (
      SELECT file, id, name FROM identifiers
      UNION ALL
      SELECT c.file, c.id,
             CASE WHEN c.type <> 'attribute' THEN q.val
                  WHEN q.val = '' THEN ''
                  ELSE q.val || '.' || coalesce(i.name, '') END
      FROM q JOIN chains c ON c.file = q.file AND c.first = q.id
      LEFT JOIN identifiers i ON i.file = c.file AND i.id = c.id + c.dc
    ) SELECT * FROM q
"""

_FRAMES = r"""
    SELECT f.fact_id, f.fact_order, r.file, r.owner, b.off + r.sc AS base, d.above AS owner_above
    FROM syntax_parented r
    JOIN syntax_facts f ON f.ast_file = r.file AND f.ast_node = r.id
    JOIN syntax_lines b ON b.path = r.path AND b.line = r.sl
    JOIN syntax_depths d ON d.file = r.file AND d.id = r.owner
    WHERE r.root
"""

_NODES = r"""
    WITH identifiers AS (
      SELECT file, id, name FROM syntax_tree WHERE type = 'identifier'
    )
    SELECT fr.fact_order, fr.fact_id, fr.fact_id || ':' || n.ordinal AS node_id,
           coalesce(fr.fact_id || ':' || p.ordinal, '') AS parent_id,
           n.ordinal, n.subtree_end,
           CASE WHEN n.root THEN 1
                ELSE a.above - fr.owner_above + 1 + CASE WHEN n.decorating THEN 1 ELSE 0 END
                     - CASE WHEN n.synthetic THEN 1 ELSE 0 END END AS depth,
           n.kind,
           CASE WHEN n.root OR n.leaf OR n.type = 'identifier' THEN n.name
                WHEN n.type = 'attribute' THEN coalesce(li.name, '')
                WHEN n.type = 'call' THEN coalesce(d.val, '')
                WHEN n.type IN ('assignment', 'augmented_assignment', 'for_statement')
                  THEN coalesce(t.name, '')
                ELSE '' END AS name,
           n.path, n.sl AS start_line, n.sc AS start_column, n.el AS end_line, n.ec AS end_column,
           ls.off + n.sc - fr.base AS byte_start,
           le.off + n.ec - ls.off - n.sc AS byte_length
    FROM syntax_parented n
    JOIN syntax_depths a ON a.file = n.file AND a.id = n.id
    JOIN syntax_frames fr ON fr.file = n.file AND fr.owner = n.owner
    LEFT JOIN syntax_numbered p ON p.file = n.file AND p.owner = n.owner AND p.key = n.parent_key
    LEFT JOIN identifiers li ON li.file = n.file AND li.id = n.eid
    LEFT JOIN syntax_dotted d ON d.file = n.file AND d.id = n.id
    LEFT JOIN syntax_targets t ON t.file = n.file AND t.id = n.id
    JOIN syntax_lines ls ON ls.path = n.path AND ls.line = n.sl
    JOIN syntax_lines le ON le.path = n.path AND le.line = n.el
"""

_CHILDREN = r"""
    SELECT n.fact_id, p.ordinal AS parent_ordinal,
           row_number() OVER (PARTITION BY n.fact_id, p.ordinal ORDER BY n.ordinal) - 1
             AS child_order,
           n.ordinal AS child_ordinal
    FROM syntax_nodes n JOIN syntax_nodes p ON p.fact_id = n.fact_id AND p.node_id = n.parent_id
"""


class Syntax(Family):
    family = "SyntaxFact"
    verified = True
    relations = {
        "facts": _fill(_FACTS),
        "lines": _LINES,
        "tree": _fill(_TREE),
        "ranks": _RANKS,
        "places": _fill(_PLACES),
        "kept": _fill(_KEPT),
        "owners": _OWNERS,
        "decorators": _DECORATORS,
        "owned": _OWNED,
        "bare": _BARE,
        "tuples": _TUPLES,
        "ends": _fill(_ENDS),
        "members": _MEMBERS,
        "links": _LINKS,
        "climb": _CLIMB,
        "swaps": _SWAPS,
        "depths": _DEPTHS,
        "numbered": _NUMBERED,
        "spanned": _SPANNED,
        "parented": _PARENTED,
        "targets": _TARGETS,
        "dotted": _DOTTED,
        "frames": _FRAMES,
        "nodes": _NODES,
        "children": _CHILDREN,
    }
