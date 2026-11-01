"""Rules over condition chains (mcmr's ALL-BRAN family)."""

from ....catalog import Example, Rule


class ValueDispatchCandidate(Rule, id="ALL-BRAN0001", applicability="review"):
    """Condition chains whose arms only test one subject for equality with distinct literals."""

    minimum_arms: int = 3

    query = """
        WITH clauses AS (
          SELECT i.file_id, i.node_id AS chain, c.node_id AS clause,
                 row_number() OVER (PARTITION BY i.file_id, i.node_id ORDER BY c.node_id) - 1
                   AS arm
          FROM core_node i
          JOIN core_file f USING (file_id)
          JOIN core_node c ON c.file_id = i.file_id
            AND (c.node_id = i.node_id OR (c.parent_id = i.node_id AND c.type = 'elif_clause'))
          WHERE i.type = 'if_statement' AND f.language = 'python'
        ), comparisons AS (
          SELECT c.file_id, c.chain, c.arm, l.type AS left_type, l.name AS left_name,
                 l.node_id AS left_node, r.type AS right_type, r.name AS right_name,
                 r.node_id AS right_node
          FROM clauses c
          JOIN core_node t ON t.file_id = c.file_id AND t.parent_id = c.clause
            AND t.sibling_index = 1
          LEFT JOIN core_node p ON p.file_id = t.file_id AND p.parent_id = t.node_id
            AND p.sibling_index = 1 AND t.type = 'parenthesized_expression'
          JOIN core_node x ON x.file_id = t.file_id AND x.node_id = coalesce(p.node_id, t.node_id)
            AND x.type = 'comparison_operator' AND x.children_count = 3
          JOIN core_node o ON o.file_id = x.file_id AND o.parent_id = x.node_id
            AND o.sibling_index = 1 AND o.type = '=='
          JOIN core_node l ON l.file_id = x.file_id AND l.parent_id = x.node_id
            AND l.sibling_index = 0
          JOIN core_node r ON r.file_id = x.file_id AND r.parent_id = x.node_id
            AND r.sibling_index = 2
        ), texts AS (
          SELECT m.file_id, m.chain, m.arm, string_agg(c.name, '' ORDER BY c.node_id) AS text
          FROM comparisons m
          JOIN core_node s ON s.file_id = m.file_id AND s.type = 'string'
            AND (s.node_id = m.right_node OR s.parent_id = m.right_node)
          JOIN core_node c ON c.file_id = s.file_id AND c.parent_id = s.node_id
            AND c.type IN ('string_content', 'escape_sequence')
          GROUP BY m.file_id, m.chain, m.arm
        ), literals AS (
          SELECT m.file_id, m.chain, m.arm,
                 CASE WHEN m.left_type IN ('identifier', 'attribute') THEN m.left_name
                      WHEN m.left_type = 'call' THEN coalesce(f.name, '') ELSE '' END AS subject,
                 CASE WHEN m.right_type IN ('string', 'concatenated_string')
                      THEN NOT regexp_matches(m.right_name, '^[a-zA-Z]*[fF]')
                      ELSE m.right_type IN ('integer', 'float', 'true', 'false', 'none',
                                            'ellipsis', 'list', 'tuple', 'dictionary', 'set')
                 END AS is_literal,
                 CASE WHEN m.right_type IN ('integer', 'float') THEN m.right_name
                      WHEN m.right_type IN ('true', 'false') THEN lower(m.right_name)
                      WHEN m.right_type IN ('string', 'concatenated_string')
                           AND NOT regexp_matches(m.right_name, '^[a-zA-Z]*[bB]')
                      THEN coalesce(t.text, '') ELSE '' END AS literal
          FROM comparisons m
          LEFT JOIN texts t ON t.file_id = m.file_id AND t.chain = m.chain AND t.arm = m.arm
          LEFT JOIN core_node f ON f.file_id = m.file_id AND f.parent_id = m.left_node
            AND f.sibling_index = 0 AND m.left_type = 'call'
            AND f.type IN ('identifier', 'attribute')
        ), subjects AS (
          SELECT l.file_id, l.chain, l.subject, n.start_line, n.start_column
          FROM literals l JOIN core_node n ON n.file_id = l.file_id AND n.node_id = l.chain
          WHERE l.arm = 0 AND l.subject <> '' AND l.is_literal
        ), arms AS (
          SELECT s.file_id, s.chain, c.arm,
                 coalesce(l.is_literal AND l.subject = s.subject, false) AS reads_subject_only,
                 CASE WHEN l.is_literal AND l.subject = s.subject THEN l.literal ELSE ''
                 END AS literal
          FROM subjects s
          JOIN clauses c ON c.file_id = s.file_id AND c.chain = s.chain
          JOIN core_node b ON b.file_id = c.file_id AND b.parent_id = c.clause AND b.type = 'block'
          LEFT JOIN literals l ON l.file_id = c.file_id AND l.chain = c.chain AND l.arm = c.arm
        ), chains AS (
          SELECT file_id, chain, count(*) AS arm_count,
                 string_agg(literal, '`, `' ORDER BY arm) AS literals
          FROM arms GROUP BY file_id, chain
          HAVING bool_and(reads_subject_only AND literal <> '')
             AND count(DISTINCT literal) = count(*) AND count(*) >= $minimum_arms
        )
        SELECT f.path, s.start_line, s.start_column, s.subject, c.arm_count, c.literals
        FROM chains c JOIN subjects s ON s.file_id = c.file_id AND s.chain = c.chain
        JOIN core_file f ON f.file_id = c.file_id
    """
    message = "`{subject}` selects among {arm_count} literal arms `{literals}`"
    change = "matched `{subject}` against its {arm_count} literal arms"
    # The chain becomes `match subject:` with one `case literal:` per arm and `else:` as `case _:`,
    # each body and each comment line between the arms one indent deeper (`rewrite-node` rows: the
    # keywords, the comparisons, the bodies and those comments; a comment on a header's line stays
    # there, and no comment or statement is lost). Left alone: a literal that is no number or
    # plain string (`True`, `None` and a list match by identity or by shape, not equality), a call
    # as the subject, a header in parentheses, a body on its header's line or indented differently
    # from arm to arm, tabs, and a string over several lines (it would be re-indented). Python
    # 3.10 reads `match`.
    fix = r"""
        WITH chains AS (
          SELECT f.finding_key, f.path, i.file_id, i.node_id AS chain, i.start_line AS first,
                 i.end_line AS last, i.start_column AS base, i.descendant_count AS size
          FROM finding f
          JOIN core_file p ON p.path = f.path
          JOIN core_node i ON i.file_id = p.file_id AND i.type = 'if_statement'
            AND i.start_line = f.start_line AND i.start_column = f.start_column
        ), arms AS (
          SELECT c.*, k.node_id AS clause, k.type AS kind, row_number() OVER w - 1 AS arm,
                 kw.node_id AS keyword, cond.node_id AS condition, cond.type AS cond_type,
                 o.node_id AS op, l.node_id AS left_node, l.type AS left_type, l.name AS subject,
                 r.node_id AS right_node, r.type AS right_type, r.name AS literal,
                 b.node_id AS block, b.start_column - c.base AS unit
          FROM chains c
          JOIN core_node k ON k.file_id = c.file_id
            AND (k.node_id = c.chain
                 OR (k.parent_id = c.chain AND k.type IN ('elif_clause', 'else_clause')))
          JOIN core_node kw ON kw.file_id = c.file_id AND kw.parent_id = k.node_id
            AND kw.sibling_index = 0
          JOIN core_node b ON b.file_id = c.file_id AND b.parent_id = k.node_id
            AND b.type = 'block'
          LEFT JOIN core_node cond ON cond.file_id = c.file_id AND cond.parent_id = k.node_id
            AND cond.sibling_index = 1 AND k.type <> 'else_clause'
          LEFT JOIN core_node o ON o.file_id = c.file_id AND o.parent_id = cond.node_id
            AND o.sibling_index = 1 AND o.type = '=='
          LEFT JOIN core_node l ON l.file_id = c.file_id AND l.parent_id = cond.node_id
            AND l.sibling_index = 0
          LEFT JOIN core_node r ON r.file_id = c.file_id AND r.parent_id = cond.node_id
            AND r.sibling_index = 2
          WINDOW w AS (PARTITION BY c.finding_key ORDER BY k.node_id)
        ), whole AS (
          SELECT a.finding_key FROM arms a
          GROUP BY a.finding_key
          HAVING count(DISTINCT a.unit) = 1 AND min(a.unit) > 0
             AND bool_and(a.kind = 'else_clause' OR (
                   a.cond_type = 'comparison_operator' AND a.op IS NOT NULL
                   AND a.left_type IN ('identifier', 'attribute')
                   AND a.subject = (SELECT s.subject FROM arms s
                                    WHERE s.finding_key = a.finding_key AND s.arm = 0)
                   AND (a.right_type IN ('integer', 'float')
                        OR (a.right_type = 'string'
                            AND NOT regexp_matches(a.literal, '^[a-zA-Z]*[fFbB]')))))
        ), spoiled AS (
          SELECT c.finding_key FROM chains c
          JOIN core_node s ON s.file_id = c.file_id AND s.type = 'string'
            AND s.start_line <> s.end_line AND s.node_id BETWEEN c.chain AND c.chain + c.size
          UNION
          SELECT c.finding_key FROM chains c
          JOIN source_lines() x ON x.path = c.path AND x.line BETWEEN c.first AND c.last
          WHERE regexp_matches(x.text, '^ *\t')
        ), settled AS (
          SELECT a.* FROM arms a JOIN whole USING (finding_key)
          WHERE a.finding_key NOT IN (SELECT finding_key FROM spoiled)
        )
        SELECT finding_key, 'rewrite-node' AS operation, path, keyword AS node,
               MAP {'OLD': [keyword]} AS captures,
               CASE kind WHEN 'if_statement' THEN 'match'
                         WHEN 'elif_clause' THEN repeat(' ', unit) || 'case'
                         ELSE repeat(' ', unit) || 'case _' END AS template
        FROM settled
        UNION ALL
        SELECT finding_key, 'rewrite-node', path, condition,
               MAP {'SUBJECT': [left_node], 'LITERAL': [right_node]},
               if(arm = 0, '$SUBJECT:' || chr(10) || repeat(' ', unit) || 'case $LITERAL',
                  '$LITERAL')
        FROM settled WHERE kind <> 'else_clause'
        UNION ALL
        SELECT s.finding_key, 'rewrite-node', s.path, s.block,
               MAP {'BODY': list(c.node_id ORDER BY c.sibling_index)},
               repeat(' ', s.unit) || '$$$BODY'
        FROM settled s JOIN core_node c ON c.file_id = s.file_id AND c.parent_id = s.block
        GROUP BY s.finding_key, s.path, s.block, s.unit
        UNION ALL
        SELECT s.finding_key, 'rewrite-node', s.path, m.node_id, MAP {'NOTE': [m.node_id]},
               repeat(' ', s.unit) || '$NOTE'
        FROM settled s
        JOIN core_node m ON m.file_id = s.file_id AND m.parent_id = s.clause AND m.type = 'comment'
        JOIN source_lines() x ON x.path = s.path AND x.line = m.start_line
        WHERE regexp_matches(substr(x.text, 1, m.start_column), '^ *$')
    """
    # The context file is the case the rule must leave alone: arms ordered by `<` are thresholds,
    # which no `match` arm can state.
    examples = (
        Example(
            bad=(
                """def label(code):
    if code == 1:
        return "low"
    elif code == 2:
        return "mid"
    elif code == 3:
        return "high"
    return "none"
"""
            ),
            good=(
                """def label(code):
    match code:
        case 1:
            return "low"
        case 2:
            return "mid"
        case 3:
            return "high"
    return "none"
"""
            ),
            context={
                "bands.py": (
                    """def band(size):
    if size < 1:
        return "tiny"
    elif size < 2:
        return "small"
    elif size < 3:
        return "medium"
    return "large"
"""
                )
            },
        ),
        Example(
            bad=(
                """class Door:
    def label(self):
        if self.state == "a":
            return 1
        elif self.state == "b":
            # the middle one
            value = 2
            return value
        elif self.state == "c":
            return 3
        else:
            return 4
"""
            ),
            good=(
                """class Door:
    def label(self):
        match self.state:
            case "a":
                return 1
            case "b":
                # the middle one
                value = 2
                return value
            case "c":
                return 3
            case _:
                return 4
"""
            ),
        ),
        Example(
            bad=(
                """def label(flag):
    if flag == 1:
        return "one"
    elif flag == 2:
        return "two"
    elif flag == True:
        return "yes"
    return "none"
"""
            ),
            good=(
                """def label(flag):
    if flag == 1:
        return "one"
    elif flag == 2:
        return "two"
    elif flag == True:
        return "yes"
    return "none"
"""
            ),
        ),
        Example(
            bad=(
                """def label(rows):
    if len(rows) == 1:
        return "one"
    elif len(rows) == 2:
        return "two"
    elif len(rows) == 3:
        return "three"
    return "many"
"""
            ),
            good=(
                """def label(rows):
    if len(rows) == 1:
        return "one"
    elif len(rows) == 2:
        return "two"
    elif len(rows) == 3:
        return "three"
    return "many"
"""
            ),
        ),
        Example(
            bad=(
                """def label(code):
    if code == 1: return "low"
    elif code == 2: return "mid"
    elif code == 3: return "high"
    return "none"
"""
            ),
            good=(
                """def label(code):
    if code == 1: return "low"
    elif code == 2: return "mid"
    elif code == 3: return "high"
    return "none"
"""
            ),
        ),
        Example(
            bad=(
                """def label(code):
    if code == 1:
        return "low"
    elif code == 2:
        return \"\"\"mid
        range\"\"\"
    elif code == 3:
        return "high"
    return "none"
"""
            ),
            good=(
                """def label(code):
    if code == 1:
        return "low"
    elif code == 2:
        return \"\"\"mid
        range\"\"\"
    elif code == 3:
        return "high"
    return "none"
"""
            ),
        ),
        Example(
            bad="def label(code):\n\tif code == 1:\n\t\treturn 1\n\telif code == 2:\n\t\treturn 2\n\telif code == 3:\n\t\treturn 3\n",
            good="def label(code):\n\tif code == 1:\n\t\treturn 1\n\telif code == 2:\n\t\treturn 2\n\telif code == 3:\n\t\treturn 3\n",
        ),
    )
