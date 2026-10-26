"""Rules over import hygiene (mcmr's PY-IMPO hygiene family)."""

from ....catalog import Example, Function, Relation, Rule


class ImportsCodeLines(Relation):
    """Every line of code (not blank, not a comment) of every source file, with the code lines
    just before and after it: what a fix needs to tell whether a statement is alone in its
    block."""

    stored = True
    query = """
        SELECT path, line, text, lag(text) OVER w AS previous_text,
               lead(text) OVER w AS following_text
        FROM source_lines()
        WHERE trimmed(text) <> '' AND NOT starts_with(trimmed(text), '#')
        WINDOW w AS (PARTITION BY path ORDER BY line)
    """


class ImportsLineGaps(Relation):
    """Every non-blank line with the numbers of the non-blank lines around it: 0 before the
    first, the line past the file's last after the last (`final`). What a fix needs to take the
    blank lines a deleted statement leaves along with it."""

    stored = True
    query = """
        WITH sizes AS (
          SELECT path, length(content) - length(replace(content, chr(10), ''))
                       + CASE WHEN content = '' OR ends_with(content, chr(10)) THEN 0 ELSE 1 END
                       AS physical
          FROM source_files
        )
        SELECT l.path, l.line, coalesce(lag(l.line) OVER w, 0) AS previous_line,
               coalesce(lead(l.line) OVER w, s.physical + 1) AS next_line,
               lead(l.line) OVER w IS NULL AS final
        FROM source_lines() l JOIN sizes s USING (path)
        WHERE trimmed(l.text) <> ''
        WINDOW w AS (PARTITION BY l.path ORDER BY l.line)
    """


class ImportsTidyStart(Function):
    """The line deleting a statement starts at: its first line, or the blank lines before it when
    nothing follows it (`previous_line` and `final` come from the gaps around its first and last
    lines)."""

    first_line: int
    previous_line: int
    final: bool

    expression = """
        CASE WHEN final AND previous_line > 0 AND first_line - previous_line > 1
             THEN previous_line + 1 ELSE first_line END
    """


class ImportsTidyEnd(Function):
    """The line deleting a statement runs up to: past its last line, or past the blank lines after
    it when blank lines or the file's start precede it (`next_line` is the next non-blank line)."""

    first_line: int
    last_line: int
    previous_line: int
    next_line: int

    expression = """
        CASE WHEN previous_line = 0 OR first_line - previous_line > 1 THEN next_line
             ELSE last_line + 1 END
    """


class ImportsBefore(Function):
    """The text of a line before a span column, which counts UTF-8 bytes."""

    text: str
    byte_column: int

    expression = "decode(array_slice(encode(text), 1, byte_column))"


class ImportsFrom(Function):
    """The text of a line from a span column on, which counts UTF-8 bytes."""

    text: str
    byte_column: int

    expression = "decode(array_slice(encode(text), byte_column + 1, strlen(text)))"


class ImportsEmptiesBlock(Function):
    """Whether deleting a statement leaves its block empty: the code line before it opens the
    block (ends in a colon) and the one after it dedents. `text` is the statement's first line,
    `following_text` the code line after its last (null at the end of the file)."""

    previous_text: str
    text: str
    following_text: str

    expression = """
        regexp_matches(previous_text, ':\\s*(#.*)?$') AND (following_text IS NULL
          OR length(following_text) - length(ltrim(following_text, ' \t'))
             < length(text) - length(ltrim(text, ' \t')))
    """


class ImportsListMembers(Relation):
    """Every member a fix may delete from a comma separated list: the bindings of an import
    statement and the items of an `__all__` assignment (`group_id` names the statement).

    Holds the member's span and its neighbors', the statement's span, and what the lines say about
    deleting them: `member_alone` (the member fills its own lines), `statement_deletable` (the
    statement fills its own lines and its block keeps another statement), `complete` (every item
    the statement states is a member, so the neighbors are the true ones), `statement_start` and
    `statement_end` (the lines deleting the statement runs from and up to, taking the blank lines
    around it that would be left over).
    """

    stored = True
    query = """
        WITH members AS (
          SELECT i.binding_id AS member_id, i.declaration_id AS group_id, i.path,
                 b.start_line::BIGINT AS bl, b.start_column::BIGINT AS bc,
                 b.end_line::BIGINT AS el, b.end_column::BIGINT AS ec,
                 d.start_line::BIGINT AS sl, d.start_column::BIGINT AS sc,
                 d.end_line::BIGINT AS sel, d.end_column::BIGINT AS sec,
                 NULL::BIGINT AS expected
          FROM import_binding_facts i
          JOIN import_binding_nodes b
            ON b.fact_id = i.fact_id AND b.node_id = i.binding_id AND b.role = 'binding'
          JOIN import_binding_nodes d
            ON d.fact_id = i.fact_id AND d.node_id = i.declaration_id AND d.role = 'declaration'
          UNION ALL
          SELECT DISTINCT ON (n.id) n.id, a.id, n.span_path,
                 n.span_start_line::BIGINT, n.span_start_column::BIGINT,
                 n.span_end_line::BIGINT, n.span_end_column::BIGINT,
                 a.span_start_line::BIGINT, a.span_start_column::BIGINT,
                 a.span_end_line::BIGINT, a.span_end_column::BIGINT,
                 len(regexp_extract_all(a.text, '"[^"\\n]*"|''[^''\\n]*'''))
          FROM export_public_export_nodes n
          JOIN module_all_declarations a ON a.span_path = n.span_path
            AND (a.span_start_line, a.span_start_column)
                <= (n.span_start_line, n.span_start_column)
            AND (n.span_end_line, n.span_end_column) <= (a.span_end_line, a.span_end_column)
        ), ordered AS (
          SELECT *, row_number() OVER w AS position,
                 lag(el) OVER w AS previous_line, lag(ec) OVER w AS previous_column,
                 lead(bl) OVER w AS next_line, lead(bc) OVER w AS next_column,
                 coalesce(count(*) OVER (PARTITION BY group_id) = expected, true) AS complete
          FROM members WINDOW w AS (PARTITION BY group_id ORDER BY bl, bc)
        )
        SELECT o.* EXCLUDE (expected),
               trimmed(imports_before(begun.text, o.bc)) = ''
                 AND regexp_matches(imports_from(ended.text, o.ec), '^\\s*,?\\s*(#.*)?$')
                 AS member_alone,
               trimmed(imports_before(opening.text, o.sc)) = ''
                 AND regexp_matches(imports_from(closing.text, o.sec), '^\\s*(#.*)?$')
                 AND NOT coalesce(imports_empties_block(
                   opening.previous_text, opening.text, closing.following_text), false)
                 AS statement_deletable,
               imports_tidy_start(o.sl, first_gap.previous_line, last_gap.final)
                 AS statement_start,
               imports_tidy_end(o.sl, o.sel, first_gap.previous_line, last_gap.next_line)
                 AS statement_end
        FROM ordered o
        LEFT JOIN imports_line_gaps() first_gap
          ON first_gap.path = o.path AND first_gap.line = o.sl
        LEFT JOIN imports_line_gaps() last_gap
          ON last_gap.path = o.path AND last_gap.line = o.sel
        LEFT JOIN imports_code_lines() begun ON begun.path = o.path AND begun.line = o.bl
        LEFT JOIN imports_code_lines() ended ON ended.path = o.path AND ended.line = o.el
        LEFT JOIN imports_code_lines() opening ON opening.path = o.path AND opening.line = o.sl
        LEFT JOIN imports_code_lines() closing ON closing.path = o.path AND closing.line = o.sel
    """


class ImportsMemberRemovals(Relation):
    """The text each member in `removed` (member ids of `imports_list_members()`) deletes from its
    list: the whole statement (on its first member) when every member goes, a member's own lines,
    else the member with the comma joining it to the next one (to the previous one for the last
    kept). `feasible` is false when the statement must go but its lines are not its own; a member
    whose deletion another already covers has null spans."""

    removed: list[str]

    query = """
        WITH touched AS (
          SELECT *, member_id IN (SELECT unnest(removed)) AS gone
          FROM imports_list_members()
          WHERE group_id IN (SELECT group_id FROM imports_list_members()
                             WHERE member_id IN (SELECT unnest(removed)))
        ), ordered AS (
          SELECT *, bool_or(NOT gone) OVER (PARTITION BY group_id ORDER BY position
                      ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS kept_after,
                 bool_and(gone) OVER (PARTITION BY group_id) AS all_gone
          FROM touched
        ), shaped AS (
          SELECT *, CASE
            WHEN NOT complete THEN NULL
            WHEN all_gone THEN CASE WHEN statement_deletable THEN 'statement' END
            WHEN member_alone THEN 'lines'
            WHEN kept_after THEN 'forward'
            WHEN previous_line IS NOT NULL THEN 'backward' END AS kind
          FROM ordered WHERE gone
        )
        SELECT member_id, group_id, path, kind IS NOT NULL AS feasible,
          CASE WHEN kind = 'statement' AND position > 1 THEN NULL ELSE
            CASE kind WHEN 'statement' THEN statement_start WHEN 'backward' THEN previous_line
                      WHEN 'lines' THEN bl WHEN 'forward' THEN bl END END AS start_line,
          CASE kind WHEN 'forward' THEN bc WHEN 'backward' THEN previous_column
                    WHEN 'statement' THEN 0 WHEN 'lines' THEN 0 END AS start_column,
          CASE kind WHEN 'statement' THEN statement_end WHEN 'lines' THEN el + 1
                    WHEN 'forward' THEN next_line WHEN 'backward' THEN el END AS end_line,
          CASE kind WHEN 'forward' THEN next_column WHEN 'backward' THEN ec
                    WHEN 'statement' THEN 0 WHEN 'lines' THEN 0 END AS end_column
        FROM shaped
    """


class ImportModuleDepth(Rule, id="PY-IMPO0005", applicability="safe"):
    """Import statements naming a module deeper than the project accepts.

    mcmr judges per binding name, so one too-deep statement reports every statement of its name."""

    maximum: int = 6

    query = """
        WITH first AS (
          SELECT *, ltrim(module, '.') AS named FROM import_binding_facts
          QUALIFY fact_order = min(fact_order) OVER (
            PARTITION BY CASE WHEN declaration_id <> '' THEN declaration_id ELSE fact_id END)
        )
        SELECT i.* EXCLUDE (named), CASE WHEN named = '' THEN 0
            ELSE len(str_split(named, '.')) END AS depth,
          f.replacement_module AS facade, f.module_node_span_path, f.module_node_span_start_line,
          f.module_node_span_start_column, f.module_node_span_end_line,
          f.module_node_span_end_column
        FROM first i LEFT JOIN imports_facades() f ON f.module_node_id = i.module_node_id
        QUALIFY max(depth) OVER (PARTITION BY i.fact_id) > $maximum
    """
    message = "named import depth is {depth} for `{fact_id}`"
    examples = (
        Example(
            bad="from pkg.a.b.c.d.e.thing import Thing\n",
            good="from pkg.a.b import Thing\n",
            path="app.py",
            context={
                "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
                "src/pkg/__init__.py": "",
                "src/pkg/a/__init__.py": "",
                "src/pkg/a/b/__init__.py": 'from .c.d.e.thing import Thing\n\n__all__ = ["Thing"]\n',
                "src/pkg/a/b/c/__init__.py": "",
                "src/pkg/a/b/c/d/__init__.py": "",
                "src/pkg/a/b/c/d/e/__init__.py": "",
                "src/pkg/a/b/c/d/e/thing.py": "class Thing:\n    pass\n",
            },
        ),
    )
    # Only the statements that are themselves too deep, and only through a facade that is within
    # the limit; `export_bypasses` proves the facade.
    fix = """
        SELECT finding_key, module_node_span_path AS path,
               module_node_span_start_line AS start_line,
               module_node_span_start_column AS start_column,
               module_node_span_end_line AS end_line, module_node_span_end_column AS end_column,
               CASE WHEN depth > $maximum AND facade IS NOT NULL
                         AND len(str_split(facade, '.')) <= $maximum THEN facade END AS text
        FROM finding
    """
