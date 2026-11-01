"""The first stages of the element stream (see `stream.py`), as common table expressions: the atoms
of every LaTeX file, their kinds, and the parts each atom contributes (prose to append, elements to
push)."""

from .reader import (
    BRACKET,
    BREAKS,
    CITATIONS,
    DISCARDED,
    ENVIRONMENT_KINDS,
    INSIDE,
    MACROS,
    MARK,
    REFERENCES,
    SECTIONS,
    THEOREMS,
)
from .sql import atom_pattern, punctuation, quoted, role, strip

WORD = r"\\[A-Za-z@]+\*?"

_LABEL_ROWS_SQL = """SELECT path, pos, 100 + i AS sub, true AS pushes, 'label' AS elem, name AS a,
                  NULL AS b, NULL AS n, line, NULL AS txt, false AS is_word
           FROM raw_labels"""
_REFERENCE_ROWS_SQL = """SELECT path, pos, i AS sub, true, 'reference', target, name, NULL, line, NULL, false
           FROM targets"""
_CITATION_ROWS_SQL = """SELECT path, pos, i AS sub, true, 'citation', target, pin, NULL, line, NULL, false
           FROM sources"""

_NAME = quoted(r"^\\([A-Za-z@]+)")

_BEGIN = quoted(r"^\\begin\*?\{([^{}]*)\}")

_END = quoted(r"^\\end\*?\{([^{}]*)\}")

_DOT = f"' ' || {MARK} || ' '"


def _in(column: str, names: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(map(quoted, names))})"


def first_group(after: str = WORD) -> str:
    """The first brace group of the atom behind `after` (a regular expression)."""
    return f"regexp_extract(atom, {quoted(rf'^{after}\{{({INSIDE})\}}')}, 1)"


def _between(opening: str, *, closing: str) -> str:
    """What sits between an opening mark and its closing one.

    An atom at the end of its file may lack the closing mark.
    """
    return (
        f"CASE WHEN ends_with(atom, {quoted(closing)}) "
        f"AND size >= {len(opening) + len(closing)} "
        f"THEN substr(atom, {len(opening) + 1}, size - {len(opening) + len(closing)}) "
        f"ELSE substr(atom, {len(opening) + 1}) END"
    )


def _tail(body: str) -> str:
    """The prose a display formula leaves behind it: a mark and the punctuation ending it."""
    ending = punctuation(body)
    return f"{_DOT} || CASE WHEN {ending} = '' THEN '' ELSE {ending} || ' ' END"


def _kind() -> str:
    """The kind of the atom: what the reader does when it meets it.

    A control word naming a class whose arguments the atom lacks is read as nothing, as mcmr's
    reader finds no group there.
    """

    def needing(kind: str) -> str:
        return f"CASE WHEN bare THEN 'nothing' ELSE '{kind}' END"

    ladder = [
        ("starts_with(atom, '%')", "'comment'"),
        ("starts_with(atom, '$$')", "'display_dollar'"),
        ("starts_with(atom, '$')", "'inline_dollar'"),
        (r"starts_with(atom, '\[')", "'display_bracket'"),
        (r"starts_with(atom, '\(')", "'inline_paren'"),
        (r"starts_with(atom, '\\')", "'row_break'"),
        (
            "name = 'begin'",
            f"CASE WHEN bare THEN 'nothing' WHEN {role('environment')} = 'math' THEN 'raw_math' "
            f"WHEN {role('environment')} = 'verbatim' THEN 'raw_verbatim' ELSE 'begin' END",
        ),
        ("name = 'end'", needing("end")),
        (_in("name", SECTIONS), needing("section")),
        ("name = 'label'", needing("label")),
        (_in("name", ("caption", "captionof")), needing("caption")),
        (_in("name", REFERENCES), needing("reference")),
        (_in("name", CITATIONS), needing("citation")),
        (_in("name", ("input", "include")), needing("include")),
        (_in("name", THEOREMS), needing("theorem")),
        ("name = 'theoremstyle'", needing("theoremstyle")),
        ("name = 'newtheoremstyle'", needing("newtheoremstyle")),
        (_in("name", MACROS), needing("macro")),
        (_in("name", ENVIRONMENT_KINDS), "'nothing'"),
        ("name = 'item'", "CASE WHEN bare THEN 'break' ELSE 'item_label' END"),
        ("name = 'bibitem'", "'break'"),
        (_in("name", BREAKS), "'break'"),
        (_in("name", ("emph", "term", "textbf", "textit", "textsc", "underline")), "'emphasis'"),
        (_in("name", DISCARDED), "'nothing'"),
        ("name IS NOT NULL", "'macro_use'"),
        ("starts_with(atom, '\\')", "'escape'"),
        ("starts_with(atom, chr(10))", "'newline'"),
        ("atom IN ('{', '}')", "'brace'"),
    ]
    branches = "\n".join(f"WHEN {when} THEN {then}" for when, then in ladder)
    return f"CASE {branches} ELSE 'word' END"


ATOMS = f"""
    cut AS (
      SELECT path, unnest(parts) AS atom, generate_subscripts(parts, 1) AS pos
      FROM (SELECT path, regexp_extract_all(content, {quoted(atom_pattern())}) AS parts
            FROM latex)
    ), measured AS (
      SELECT *, length(atom) AS size, length(atom) - length(replace(atom, chr(10), '')) AS breaks
      FROM cut
    ), atoms AS (
      SELECT path, pos, atom, size,
             (coalesce(sum(size) OVER walk, 0) - size)::BIGINT AS start_offset,
             (1 + coalesce(sum(breaks) OVER walk, 0) - breaks)::BIGINT AS line,
             (1 + coalesce(sum(breaks) OVER walk, 0))::BIGINT AS end_line,
             lead(atom, 1) OVER walk AS next_one, lead(atom, 2) OVER walk AS next_two,
             CASE WHEN regexp_matches(atom, {quoted(r"^\\[A-Za-z@]")})
                  THEN regexp_extract(atom, {_NAME}, 1) END AS name,
             regexp_matches(atom, {quoted(r"^\\[A-Za-z@]+\*?$")}) AS bare,
             {strip(f"regexp_extract(atom, {_BEGIN}, 1)")} AS environment
      FROM measured
      WINDOW walk AS (PARTITION BY path ORDER BY pos)
    ), typed AS MATERIALIZED (
      SELECT *, {_kind()} AS kind FROM atoms
    )
"""


def _part(
    kind: str,
    sub: int,
    *,
    elem: str | None = None,
    where: str = "true",
    source: str = "typed",
    **columns: str,
) -> str:
    """One piece the atoms of `kind` contribute: an element when `elem` names one, else prose.

    `columns` gives the SQL of the columns a, b, n, line and text; a word is the `word` kind.
    """
    return (
        f"SELECT path, pos, {sub} AS sub, {'true' if elem else 'false'} AS pushes, "
        f"{elem or 'NULL'} AS elem, {columns.get('a', 'NULL')} AS a, "
        f"{columns.get('b', 'NULL')} AS b, {columns.get('n', 'NULL')} AS n, "
        f"{columns.get('line', 'line')} AS line, {columns.get('text', 'NULL')} AS txt, "
        f"{str(kind == 'word').lower()} AS is_word "
        f"FROM {source} WHERE kind = '{kind}' AND {where}"
    )


def atom_parts() -> str:
    """Every piece the atoms contribute, as the rows of one UNION ALL."""
    body = (
        rf"regexp_replace(regexp_replace(atom, "
        rf"{quoted(rf'^\\begin\*?\{{[^{{}}]*\}}(?:{BRACKET})*')}, ''), "
        rf"{quoted(r'\\end\*?\{[^{}]*\}$')}, '')"
    )
    blank_after_comment = (
        "ends_with(atom, chr(10)) AND (starts_with(next_one, chr(10)) OR "
        r"(regexp_matches(next_one, '^[ \t\r]+$') AND starts_with(next_two, chr(10))))"
    )
    parts = [
        _part("word", 0, text="atom"),
        _part("newline", 0, elem="'break'", where="contains(substr(atom, 2), chr(10))"),
        _part("newline", 0, text="' '", where="NOT contains(substr(atom, 2), chr(10))"),
        _part("escape", 0, text="substr(atom, 2)"),
        _part("comment", 0, elem="'break'", line="end_line", where=blank_after_comment),
        _part("row_break", 0, elem="'rowbreak'"),
        _part("inline_dollar", 0, text=_DOT),
        _part(
            "inline_dollar",
            1,
            elem="'math'",
            n="0",
            a=rf"regexp_extract(atom, {quoted(r'^\$([^$]*)')}, 1)",
        ),
        _part("inline_paren", 0, text=_DOT),
        _part("inline_paren", 1, elem="'math'", n="0", a=_between(r"\(", closing=r"\)")),
        _part("display_dollar", 0, elem="'math'", n="1", a=_between("$$", closing="$$")),
        _part("display_dollar", 1, text=_tail(_between("$$", closing="$$"))),
        _part("display_bracket", 0, elem="'math'", n="1", a=_between(r"\[", closing=r"\]")),
        _part("display_bracket", 1, text=_tail(_between(r"\[", closing=r"\]"))),
        _part("raw_math", 0, elem="'open'", a="environment"),
        _part("raw_math", 500, elem="'math'", n="1", a=body),
        _part("raw_math", 501, text=_tail(body)),
        _part("raw_math", 999, elem="'close'", a="environment", line="end_line"),
        _part("raw_verbatim", 0, elem="'open'", a="environment"),
        _part("raw_verbatim", 999, elem="'close'", a="environment", line="end_line"),
        _part("begin", 0, elem="'bodystart'", where="environment = 'document'"),
        _part("begin", 1, elem="'open'", a="environment"),
        _part("end", 0, elem="'close'", a=strip(f"regexp_extract(atom, {_END}, 1)")),
        _part(
            "section",
            0,
            elem="'section'",
            a=strip(first_group(rf"{WORD}(?:{BRACKET})?")),
            n=f"array_position({list(SECTIONS)}, name) - 1",
        ),
        _part("label", 0, elem="'label'", a=strip(first_group())),
        _part(
            "caption",
            0,
            elem="'caption'",
            a=strip(f"regexp_extract(atom, {quoted(rf'\{{({INSIDE})\}}$')}, 1)"),
        ),
        _part("include", 0, elem="'include'", a=strip(first_group())),
        _part("item_label", 0, elem="'break'"),
        _part(
            "item_label",
            1,
            elem="'item_label'",
            a=rf"regexp_extract(atom, {quoted(r'(?s)^\\item\*?\[(.*)\]$')}, 1)",
        ),
        _part("break", 0, elem="'break'"),
        _part("macro_use", 0, elem="'macro_use'", a="name"),
        _part("macro_use", 1, text="' '"),
        _part(
            "theorem",
            0,
            elem="'statement_kind'",
            a=strip(first_group(rf"{WORD}(?:{BRACKET})?")),
            n="owes_proof",
            source="theorems",
        ),
        _part(
            "macro",
            0,
            elem="'macro'",
            a="declared",
            b="replacement",
            source="macros",
            where="declared <> '' AND (NOT has_arguments OR textual)",
        ),
        _part(
            "macro",
            1,
            elem="'heading_macro'",
            a="declared",
            b="heading_title",
            source="macros",
            n=f"(array_position({list(SECTIONS)}, heading_command) - 1) * 2 + (name <> 'providecommand')::INTEGER",
            where="declared <> '' AND NOT has_arguments AND heading_command <> ''",
        ),
        _part("emphasis", 0, elem="'emphasis'", a="marked", b="name", source="emphasized"),
    ]
    explicit = [
        # labels inside a raw environment, one per `\label` the body spells
        _LABEL_ROWS_SQL,
        # one reference per target a command names, then the mark the reference leaves in prose
        _REFERENCE_ROWS_SQL,
        f"""SELECT DISTINCT path, pos, 100000, false, NULL, NULL, NULL, NULL, line, {_DOT}, false
            FROM targets""",
        _CITATION_ROWS_SQL,
        f"""SELECT DISTINCT path, pos, 100000, false, NULL, NULL, NULL, NULL, line, {_DOT}, false
            FROM sources""",
    ]
    return "\nUNION ALL\n".join(parts + explicit)
