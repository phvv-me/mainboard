"""`core_text` (the comments and string literals of a file) and the functions that read them."""

from ...catalog import Function


class SyntaxUnescaped(Function):
    """The value of a non-raw string body: backslash escapes decoded as Python does (a named
    `\\N{...}` character stays as written)."""

    body: str

    expression = r"""
        CASE WHEN NOT contains(body, chr(92)) THEN body ELSE array_to_string(list_transform(
          regexp_extract_all(body, '(?s)\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}'
                                   || '|[0-7]{1,3}|N\{[^}]*\}|\r?\n|.)|[^\\]+'),
          lambda t: CASE
            WHEN t[1] <> chr(92) THEN t
            WHEN t[2] = 'n' THEN chr(10) WHEN t[2] = 't' THEN chr(9) WHEN t[2] = 'r' THEN chr(13)
            WHEN t[2] = 'a' THEN chr(7) WHEN t[2] = 'b' THEN chr(8) WHEN t[2] = 'f' THEN chr(12)
            WHEN t[2] = 'v' THEN chr(11)
            WHEN t[2] IN ('x', 'u', 'U') THEN coalesce(try(chr(('0x' || t[3:])::INTEGER)), t)
            WHEN t[2] IN ('0', '1', '2', '3', '4', '5', '6', '7')
              THEN chr(list_reduce(list_transform(string_split(t[2:], ''), lambda d: d::INTEGER),
                                   lambda a, b: a * 8 + b))
            WHEN t[2] IN (chr(10), chr(13)) THEN ''
            WHEN t[2] = 'N' THEN t
            WHEN t[2] IN (chr(92), chr(39), chr(34)) THEN t[2]
            ELSE t END), '') END
    """


class SyntaxSlice(Function):
    """The text of a file between two zero-based UTF-8 byte positions (lines from 1)."""

    content: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    expression = r"""
        list_transform([list_slice(string_split(content, chr(10)), start_line, end_line)],
          lambda block: decode(array_slice(encode(array_to_string(block, chr(10))),
            start_column + 1,
            strlen(array_to_string(list_slice(block, 1, len(block) - 1), chr(10)))
              + (len(block) > 1)::INTEGER + end_column)))[1]
    """


TEXT = r"""
    WITH strings AS (
      SELECT n.file_id, n.node_id, n.parent_id, n.sibling_index AS sib, n.type, n.name AS text,
             n.start_line, n.start_column, n.end_line, n.end_column
      FROM core_node n JOIN core_file f USING (file_id)
      WHERE f.language = 'python' AND n.type IN ('string', 'concatenated_string')
    ), fragments AS (
      SELECT s.*, p.type AS parent_type, lower(regexp_extract(s.text, '^[A-Za-z]*')) AS prefix
      FROM strings s LEFT JOIN core_node p ON p.file_id = s.file_id AND p.node_id = s.parent_id
      WHERE s.type = 'string'
    ), valued AS (
      SELECT *, NOT regexp_matches(prefix, '[fbt]') AS plain,
             CASE WHEN contains(prefix, 'r') THEN body ELSE syntax_unescaped(body) END AS value
      FROM (SELECT *, substr(text, strlen(prefix) + quote + 1,
                             length(text) - strlen(prefix) - 2 * quote) AS body
            FROM (SELECT *, CASE WHEN regexp_matches(substr(text, strlen(prefix) + 1),
                                                    '^(\x22{3}|\x27{3})') THEN 3 ELSE 1 END
                              AS quote FROM fragments))
    ), units AS (
      SELECT file_id, node_id, parent_id, parent_type, text, start_line, start_column, end_line,
             end_column, 1 AS fragments, plain, value
      FROM valued WHERE parent_type IS DISTINCT FROM 'concatenated_string'
      UNION ALL
      SELECT c.file_id, c.node_id, c.parent_id, p.type, c.text, c.start_line, c.start_column,
             c.end_line, c.end_column, count(*), bool_and(v.plain),
             string_agg(v.value, '' ORDER BY v.sib)
      FROM strings c JOIN valued v ON v.file_id = c.file_id AND v.parent_id = c.node_id
      LEFT JOIN core_node p ON p.file_id = c.file_id AND p.node_id = c.parent_id
      WHERE c.type = 'concatenated_string'
      GROUP BY c.file_id, c.node_id, c.parent_id, p.type, c.text, c.start_line, c.start_column,
               c.end_line, c.end_column
    ), repeated AS (
      SELECT u.file_id, u.node_id, b.node_id AS repeat_id,
             try_cast(replace(i.name, '_', '') AS BIGINT) AS repeat_count
      FROM units u
      JOIN core_node b ON b.file_id = u.file_id AND b.node_id = u.parent_id
                      AND b.type = 'binary_operator'
      JOIN core_node i ON i.file_id = b.file_id AND i.parent_id = b.node_id AND i.type = 'integer'
      JOIN core_node o ON o.file_id = b.file_id AND o.parent_id = b.node_id AND o.type = '*'
                      AND o.sibling_index = 1
      WHERE u.plain AND u.value <> ''
    )
    SELECT f.file_id, f.path, f.language, f.parses, 'string' AS kind, u.node_id AS text_id,
           u.parent_id, u.parent_type, u.start_line, u.start_column, u.end_line, u.end_column,
           u.text, u.fragments, u.plain, u.value, r.repeat_id, r.repeat_count
    FROM units u JOIN core_file f USING (file_id)
    LEFT JOIN repeated r ON r.file_id = u.file_id AND r.node_id = u.node_id
    UNION ALL BY NAME
    SELECT f.file_id, f.path, f.language, f.parses, 'comment' AS kind, n.node_id AS text_id,
           n.parent_id, p.type AS parent_type, n.start_line, n.start_column, n.end_line,
           n.end_column, n.name AS text
    FROM core_node n JOIN core_file f USING (file_id)
    LEFT JOIN core_node p ON p.file_id = n.file_id AND p.node_id = n.parent_id
    WHERE n.semantic_type = 'METADATA_COMMENT' AND n.flags % 2 = 0
"""
