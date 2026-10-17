"""SQL expressions every rule may call, installed as scalar macros."""

from ..catalog import Function, Relation


class Trimmed(Function):
    """`text` without the whitespace around it (`trim` alone strips only spaces)."""

    text: str

    expression = "trim(text, ' \t\n\r\x0b\x0c')"


class Plural(Function):
    """`1 line`, `2 lines`: `amount` and its noun, inflected."""

    amount: int
    noun: str

    expression = "amount || ' ' || noun || CASE WHEN amount = 1 THEN '' ELSE 's' END"


class NodeText(Function):
    """A syntax node's text, cut from its definition's `source` by byte offsets."""

    source: str
    byte_start: int
    byte_length: int

    expression = "decode(array_slice(encode(source), byte_start + 1, byte_start + byte_length))"


class SourceLines(Relation):
    """Every line of every source file a fact names: its number (from 1) and its text without the
    line break, for fixes that need indentation, blank lines or whole lines."""

    stored = True
    query = """
        SELECT path, generate_subscripts(lines, 1) AS line, unnest(lines) AS text
        FROM (SELECT path, string_split(content, chr(10)) AS lines FROM source_files)
    """


class TypeCheckingBlocks(Relation):
    """The lines of every `if TYPE_CHECKING:` body: what only a type checker reads, the way it
    reads a stub, and the `else` that follows (what runs) not among them."""

    stored = True
    query = """
        SELECT f.path, b.start_line, b.end_line
        FROM ast_nodes i
        JOIN ast_files f ON f.fact_id = i.fact_id AND f.language = 'python'
        JOIN ast_nodes c ON c.fact_id = i.fact_id AND c.parent_id = i.node_id
          AND c.sibling_index = 1 AND c.name IN ('TYPE_CHECKING', 'typing.TYPE_CHECKING')
        JOIN ast_nodes b ON b.fact_id = i.fact_id AND b.parent_id = i.node_id
          AND b.type = 'block' AND b.sibling_index = 2
        WHERE i.type = 'if_statement'
    """
