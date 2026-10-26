"""Rules over how control moves through a body (mcmr's ALL-CONT flow family)."""

from ....catalog import Rule

_INERT_OPERATOR = r"[=!<>]=|\bis\b|\bin\b|\bnot\b"
_ASSIGNMENT_OPERATOR = r"(?:\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=)"
_SPACE = " \t\n\r\x0b\x0c"


class StatementWithoutEffect(Rule, id="ALL-CONT0002"):
    """Statements that compute a value and then throw it away."""

    inert_kinds: tuple[str, ...] = ("name", "member", "literal", "collection", "operation")

    query = f"""
        WITH kinds(type, kind) AS (VALUES
          ('identifier', 'name'), ('member_expression', 'member'), ('attribute', 'member'),
          ('integer', 'literal'), ('float', 'literal'), ('true', 'literal'), ('false', 'literal'),
          ('none', 'literal'), ('number', 'literal'), ('null', 'literal'),
          ('list', 'collection'), ('set', 'collection'), ('tuple', 'collection'),
          ('dictionary', 'collection'), ('expression_list', 'collection'),
          ('pattern_list', 'collection'), ('array', 'collection'), ('object', 'collection'),
          ('binary_operator', 'operation'), ('boolean_operator', 'operation'),
          ('comparison_operator', 'operation'), ('unary_operator', 'operation'),
          ('not_operator', 'operation'), ('binary_expression', 'operation'),
          ('unary_expression', 'operation')
        ), discarded AS (
          SELECT f.path, k.kind, c.start_line, c.start_column, c.end_line, c.end_column
          FROM core_node e
          JOIN core_file f USING (file_id)
          JOIN core_node c ON c.file_id = e.file_id AND c.parent_id = e.node_id
          JOIN kinds k ON k.type = c.type
          WHERE e.type = 'expression_statement' AND f.parses
            AND list_contains($inert_kinds::VARCHAR[], k.kind)
            AND (f.language = 'python'
                 AND (c.start_line, c.start_column, c.end_line, c.end_column)
                   = (e.start_line, e.start_column, e.end_line, e.end_column)
                 OR f.language <> 'python'
                 AND (c.start_line, c.start_column) = (e.start_line, e.start_column)
                 AND NOT EXISTS (SELECT 1 FROM core_node o
                                 WHERE o.file_id = e.file_id AND o.parent_id = e.node_id
                                   AND o.node_id <> c.node_id
                                   AND o.semantic_type NOT IN ('PARSER_DELIMITER',
                                                               'METADATA_COMMENT')))
        ), tuples AS (
          SELECT f.path, 'collection' AS kind, e.start_line, e.start_column, e.end_line,
                 e.end_column
          FROM core_node e
          JOIN core_file f USING (file_id)
          WHERE e.type = 'expression_statement' AND f.language = 'python' AND f.parses
            AND list_contains($inert_kinds::VARCHAR[], 'collection')
            AND (SELECT count(*) FROM core_node c
                 WHERE c.file_id = e.file_id AND c.parent_id = e.node_id
                   AND c.type <> 'comment') >= 2
        ), texted AS (
          SELECT d.path, d.kind, d.start_line, d.start_column,
                 syntax_slice(s.content, d.start_line, d.start_column, d.end_line,
                              d.end_column) AS text
          FROM (SELECT * FROM discarded UNION ALL SELECT * FROM tuples) d
          JOIN source_files s USING (path)
        )
        SELECT path, start_line, start_column, kind, trim(text, '{_SPACE}') AS text FROM texted
        WHERE kind <> 'operation'
           OR (NOT regexp_matches(text, $assignment) AND regexp_matches(text, $inert))
    """
    message = "{kind} expression `{text}` is discarded"

    constants = {"assignment": _ASSIGNMENT_OPERATOR, "inert": _INERT_OPERATOR}


class DeeplyNestedBody(Rule, id="ALL-CONT0004"):
    """Declarations nesting bodies deeper than `maximum_depth`."""

    maximum_depth: int = 3
    body_types: tuple[str, ...] = (
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "try_statement",
        "with_statement",
    )

    query = """
        WITH bodies AS (
          SELECT n.file_id, n.node_id, n.descendant_count AS size, n.owner_id
          FROM core_node n JOIN core_node p ON p.file_id = n.file_id AND p.node_id = n.parent_id
          WHERE list_contains($body_types::VARCHAR[], n.type) AND p.type <> 'else_clause'
        ), nesting AS (
          SELECT t.file_id, t.owner_id, count(*) AS depth
          FROM bodies t JOIN bodies a ON a.file_id = t.file_id AND a.owner_id = t.owner_id
            AND a.node_id <= t.node_id AND t.node_id <= a.node_id + a.size
          GROUP BY t.file_id, t.owner_id, t.node_id
        )
        SELECT d.path, d.start_line, d.start_column,
          '`' || d.qualname || '` nests bodies ' || max(n.depth) || ' levels deep, past the '
          || 'ceiling of ' || $maximum_depth AS message
        FROM nesting n JOIN core_def d ON d.file_id = n.file_id AND d.def_id = n.owner_id
        WHERE d.parses
        GROUP BY d.file_id, d.def_id, d.path, d.start_line, d.start_column, d.qualname
        HAVING max(n.depth) > $maximum_depth
    """
