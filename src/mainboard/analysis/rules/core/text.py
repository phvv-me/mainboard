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
