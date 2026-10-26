"""Rules over condition chains (mcmr's ALL-BRAN family)."""

from ....catalog import Rule


class ValueDispatchCandidate(Rule, id="ALL-BRAN0001"):
    """Condition chains whose arms only compare one subject against distinct literals."""

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
