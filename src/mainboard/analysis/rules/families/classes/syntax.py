"""What the syntax trees say about each class: its header, decorators, members and fields.

ClassFact's relations, in dependency order, each reading the earlier ones by their view names
(`class_nodes` for the relation `nodes`). Every class and method is named by a stable id
(`classes:<path>:class:<n>`), and a node by its `node_id` in its file; `fact_id` is the file's.
mcmr reads only the statements directly in a class body (a method under an `if` is not a method)
and a method's statements at any depth, which the relations below follow.
"""

from ....catalog import Function


class ClassesText(Function):
    """The source text of a span, cut from the file's `lines` by UTF-8 byte columns."""

    lines: list[str]
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    expression = """array_to_string(list_concat(
        [decode(array_slice(encode(lines[start_line]), start_column + 1,
                            CASE WHEN start_line = end_line THEN end_column
                                 ELSE strlen(lines[start_line]) END))],
        lines[start_line + 1:end_line - 1],
        CASE WHEN end_line > start_line
             THEN [decode(array_slice(encode(lines[end_line]), 1, end_column))] ELSE [] END),
      chr(10))"""


class ClassesLastName(Function):
    """The last dotted segment of a decorator or base written as text, without call or subscript:
    `abc.abstractmethod` and `register(x)` read `abstractmethod` and `register`."""

    text: str

    expression = (
        r"""regexp_extract(rtrim(trim(regexp_extract(text, '^[^\[\(]*')), '.'), '[^.]*$')"""
    )
