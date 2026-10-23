"""mcmr's CommentFact, derived from sitting_duck's comment nodes.

A file's comments are grouped as mcmr groups them: comments on neighboring lines of one kind form
a group, and a directive (an instruction to a tool: `noqa`, `eslint-disable`, `clippy::`), a
documentation comment (`///`, `//!`, `/**`) and a plain one never join. A group's size is the sum
of its comments' (raw bytes, whitespace tokens, lines spanned); its text is the comment bodies
with their markers stripped; it is commented-out code (`parses_as_source`) when its body, handed
to the language's parser, is source: tree-sitter in SQL (`comment_parses`), with Python held to
what `ast.parse` accepts, after mcmr's own prefilter on the punctuation code cannot do without.

Beyond mcmr's languages (Python, Rust, TypeScript and JavaScript, C, C++, CUDA), the same groups
exist for Go, TOML, shell and the HTML comments of Markdown. They are never commented-out code
(no parser is asked), so only the length and work-marker rules read them.

Differences from the kernel, kept on purpose: a group does not state `preceding_source` and
`following_source`, which no rule reads; CUDA bodies are parsed as C++ (a commented-out kernel
launch `<<<...>>>` does not parse).
"""

from ...catalog import Family, Function


class TreeClean(Function):
    """Whether tree-sitter parses `code` in `grammar` with no error: no ERROR node and no token it
    had to insert (a leaf with no text)."""

    code: str
    grammar: str

    expression = """
        len(list_filter(parse_ast_list(code, grammar),
                        lambda n: n.type = 'ERROR' OR (n.children_count = 0 AND n.peek IS NULL)))
          = 0
    """


class PythonSource(Function):
    """Whether `body` is Python 3 source, as `ast.parse` would say: a clean tree-sitter parse that
    breaks none of the rules CPython adds to the grammar tree-sitter knows (Python 2's print, exec
    and backquotes, leading-zero integers, an unparenthesized `:=` statement, `as` outside
    `with`, `except`, `import` and `case`, an annotated tuple or list, and two statements on one
    line without `;`), with `python_annotated` for annotations tree-sitter cannot read. On the
    51,780 punctuated comment bodies of this workspace (2026-10-23) the two agree with
    `ast.parse` on every one."""

    body: str

    expression = r"""
        tree_clean(body, 'python') AND (
          WITH tree AS (SELECT parse_ast_list(body, 'python') AS n)
          SELECT len(list_filter(n, lambda x:
              x.type IN ('print_statement', 'exec_statement')
              OR (x.type = 'string_start' AND x.peek = '`')
              OR (x.type = 'integer' AND regexp_matches(x.peek, '^0[0-9_]*[1-9]'))
              OR (x.type = 'constrained_type'
                  AND n[n[x.parent_id + 1].parent_id + 1].type IS DISTINCT FROM 'type_parameter')
              OR (x.type = 'named_expression'
                  AND n[x.parent_id + 1].type = 'expression_statement')
              OR (x.type = 'as_pattern' AND n[x.parent_id + 1].type NOT IN
                    ('with_item', 'except_clause', 'except_group_clause', 'case_pattern',
                     'case_clause'))
              OR (x.type = 'assignment'
                  AND (n[x.node_id + 2].type IN ('pattern_list', 'expression_list', 'list',
                                                 'tuple', 'list_pattern')
                       OR (n[x.node_id + 2].type = 'tuple_pattern'
                           AND len(list_filter(n, lambda y: y.parent_id = x.node_id + 1
                                                  AND y.type = ',')) > 0))
                  AND len(list_filter(n, lambda y: y.parent_id = x.node_id AND y.type = ':')) > 0)
              OR (x.type NOT IN (';', 'comment')
                  AND NOT regexp_matches(x.peek, '^[\[(]')
                  AND list_contains(['module', 'block'], n[x.parent_id + 1].type)
                  AND len(list_filter(n, lambda y: y.parent_id = x.parent_id
                                         AND y.sibling_index < x.sibling_index
                                         AND y.end_line = x.start_line
                                         AND y.type NOT IN (';', 'comment'))) > 0
                  AND len(list_filter(n, lambda y: y.parent_id = x.parent_id AND y.type = ';'
                                         AND y.start_line = x.start_line)) = 0))) = 0
          FROM tree)
    """


class PythonAnnotated(Function):
    """Whether `body` is an annotated name (`Pattern: start[-end][:step]`), which CPython reads
    as any expression after the colon while tree-sitter's annotation grammar takes types only:
    a dotted name that is no keyword, a colon, and one expression (a bare tuple is refused)."""

    body: str

    expression = r"""
        regexp_matches(body, '^[A-Za-z_][A-Za-z0-9_.]*\s*:[^=]+$')
        AND NOT regexp_matches(body, '^(False|None|True|and|as|assert|async|await|break|class'
                                     || '|continue|def|del|elif|else|except|finally|for|from'
                                     || '|global|if|import|in|is|lambda|nonlocal|not|or|pass'
                                     || '|raise|return|try|while|with|yield)\s*:')
        AND (WITH wrapped AS (SELECT '_ = (' || regexp_extract(body, ':\s*(.*)$', 1) || ')' AS code)
             SELECT python_source(code)
                    AND len(list_filter(parse_ast_list(code, 'python'),
                                        lambda n: n.parent_id = 2 AND n.type = 'tuple')) = 0
             FROM wrapped)
    """


class CommentParses(Function):
    """Whether a comment's `body` is source in its `language`: Python as `ast.parse` says,
    a C-family body bare or inside a function (statements only parse there), CUDA as C++; never
    for a language no parser is asked about."""

    language: str
    body: str

    expression = r"""
        CASE
          WHEN language = 'python' THEN python_source(body) OR python_annotated(body)
          WHEN language IN ('rust', 'typescript', 'javascript', 'c', 'cpp', 'cuda') THEN
            tree_clean(CASE language WHEN 'rust' THEN 'fn mcmr_probe() {'
                                     WHEN 'typescript' THEN 'function mcmrProbe() {'
                                     WHEN 'javascript' THEN 'function mcmrProbe() {'
                                     ELSE 'void mcmr_probe() {' END
                         || chr(10) || body || chr(10) || '}',
                       CASE language WHEN 'cuda' THEN 'cpp' ELSE language END)
            OR tree_clean(body, CASE language WHEN 'cuda' THEN 'cpp' ELSE language END)
          ELSE false END
    """


class CommentBody(Function):
    """What a comment says without the markers that made it a comment: after `#` and one space
    (`hash`), between `<!--` and `-->` (`html`), or by the C family's rules (`slash`: leading
    slashes and an `!`, or the `*` gutter of a block)."""

    text: str
    style: str

    expression = r"""
        CASE style
          WHEN 'hash' THEN regexp_replace(regexp_replace(text, '^#+', ''), '^ |\s+$', '', 'g')
          WHEN 'html' THEN regexp_replace(regexp_replace(text, '^<!--|-->\s*$', '', 'g'),
                                          '^\s+|\s+$', '', 'g')
          ELSE CASE WHEN starts_with(text, '/*') THEN regexp_replace(
                 array_to_string(list_transform(
                   string_split(regexp_replace(substr(text, 3), '(\*/)+$', ''), chr(10)),
                   lambda line: regexp_replace(
                     ltrim(regexp_replace(line, '^\s+|\s+$', '', 'g'), '*'), '^\s+', '')),
                 chr(10)), '^\s+|\s+$', '', 'g')
               ELSE regexp_replace(CASE WHEN starts_with(ltrim(text, '/'), '!')
                                        THEN substr(ltrim(text, '/'), 2) ELSE ltrim(text, '/') END,
                                   '^\s+|\s+$', '', 'g') END
        END
    """


class CommentTokens(Function):
    """The whitespace-separated words of a text."""

    text: str

    expression = r"len(list_filter(regexp_split_to_array(text, '\s+'), lambda word: word <> ''))"


class Comment(Family):
    family = "CommentFact"
    verified = True
    relations = {
        "facts": r"""
            SELECT row_number() OVER (ORDER BY path) - 1 AS fact_order,
                   'comments:' || path AS fact_id, path, 1 AS start_line, 0 AS start_column,
                   1 AS end_line, 0 AS end_column, language
            FROM ast_files
            WHERE language IN ('python', 'rust', 'typescript', 'javascript', 'c', 'cpp', 'cuda',
                               'go', 'toml', 'bash', 'markdown')
              AND NOT (language = 'python' AND fact_id IN (
                SELECT fact_id FROM ast_nodes
                WHERE type = 'ERROR' OR (children_count = 0 AND start_line = end_line
                                         AND start_column = end_column AND type <> 'module')))
        """,
        "lines": r"""
            SELECT f.fact_id, f.path, generate_subscripts(a.lines, 1) AS line,
                   unnest(a.lines) AS text
            FROM comment_facts f
            JOIN (SELECT path, string_split(content, chr(10)) AS lines FROM source_files) a
              USING (path)
            WHERE f.path IN (SELECT DISTINCT n.path FROM ast_nodes n
                             WHERE n.type IN ('comment', 'line_comment', 'block_comment',
                                              'html_block'))
        """,
        "comments": r"""
            WITH dialects(language, style, markers, punctuation, documents) AS (VALUES
              ('python', 'hash', ['noqa', 'type:', 'pragma', 'ruff:', 'mypy:', 'pyright:', 'fmt:',
                                  'isort:'], ['=', '(', ':'], false),
              ('toml', 'hash', ['taplo:'], []::VARCHAR[], false),
              ('bash', 'hash', ['shellcheck', 'shfmt'], []::VARCHAR[], false),
              ('markdown', 'html', []::VARCHAR[], []::VARCHAR[], false),
              ('rust', 'slash', ['rustfmt', 'clippy', 'tarpaulin', 'grcov', 'coverage:',
                                 'codecov', 'cspell', 'cbindgen'], ['=', '(', ';', '{'], true),
              ('typescript', 'slash', ['@ts-ignore', '@ts-expect-error', 'eslint',
                                       'prettier-ignore', 'istanbul ignore', 'c8 ignore',
                                       'biome-ignore', 'deno-lint-ignore', 'oxlint-disable'],
                             ['=', '(', ';', '{'], true),
              ('javascript', 'slash', ['@ts-ignore', '@ts-expect-error', 'eslint',
                                       'prettier-ignore', 'istanbul ignore', 'c8 ignore',
                                       'biome-ignore', 'deno-lint-ignore', 'oxlint-disable'],
                             ['=', '(', ';', '{'], true),
              ('c', 'slash', ['nolint', 'clang-format', 'cppcheck-suppress', 'iwyu pragma',
                              'coverity', 'lcov_excl', 'codecov', 'cspell', 'nosonar'],
                     [';', '{'], true),
              ('cpp', 'slash', ['nolint', 'clang-format', 'cppcheck-suppress', 'iwyu pragma',
                                'coverity', 'lcov_excl', 'codecov', 'cspell', 'nosonar'],
                     [';', '{'], true),
              ('cuda', 'slash', ['nolint', 'clang-format', 'cppcheck-suppress', 'iwyu pragma',
                                 'coverity', 'lcov_excl', 'codecov', 'cspell', 'nosonar'],
                     [';', '{'], true),
              ('go', 'slash', ['nolint', 'go:'], []::VARCHAR[], true)
            ), found AS (
              SELECT f.fact_id, f.fact_order, f.path, f.language, n.start_line, n.start_column,
                     n.end_line, n.end_column
              FROM ast_nodes n JOIN comment_facts f ON f.path = n.path
              WHERE n.type IN ('comment', 'line_comment', 'block_comment', 'html_block')
            ), cut AS (
              SELECT c.*, string_agg(
                       decode(array_slice(encode(l.text),
                                          CASE WHEN l.line = c.start_line
                                               THEN c.start_column + 1 ELSE 1 END,
                                          CASE WHEN l.line = c.end_line
                                               THEN c.end_column ELSE strlen(l.text) END)),
                       chr(10) ORDER BY l.line) AS cut_text
              FROM found c JOIN comment_lines l ON l.fact_id = c.fact_id
                                               AND l.line BETWEEN c.start_line AND c.end_line
              GROUP BY ALL
            ), placed AS (
              SELECT *, rtrim(cut_text, chr(10) || chr(13)) AS text FROM cut
              WHERE language <> 'markdown' OR starts_with(cut_text, '<!--')
            ), measured AS (
              SELECT p.fact_id, p.fact_order, p.path, p.language, p.text, d.style, d.markers,
                     d.punctuation, p.start_line, p.start_column,
                     p.start_line + len(string_split(p.text, chr(10))) - 1 AS end_line,
                     CASE WHEN contains(p.text, chr(10))
                          THEN strlen(regexp_extract(p.text, '[^\n]*$'))
                          ELSE p.start_column + strlen(p.text) END AS end_column,
                     comment_body(p.text, d.style) AS body,
                     d.documents AND regexp_matches(p.text, '^(///|//!|/\*\*|/\*!)')
                       AS is_documentation
              FROM placed p JOIN dialects d USING (language)
            )
            SELECT fact_order, fact_id, path, language, style, punctuation, text, body,
                   start_line, start_column, end_line, end_column,
                   end_line - start_line + 1 AS line_count,
                   len(list_filter(markers, lambda m: starts_with(lower(body), m))) > 0
                     AS is_directive,
                   is_documentation,
                   row_number() OVER (PARTITION BY fact_id ORDER BY start_line, start_column) - 1
                     AS position
            FROM measured
        """,
        "groups": r"""
            WITH ordered AS (
              SELECT *, lag(end_line) OVER ahead AS previous_end,
                     lag(is_directive) OVER ahead AS previous_directive,
                     lag(is_documentation) OVER ahead AS previous_documentation
              FROM comment_comments WINDOW ahead AS (PARTITION BY fact_id ORDER BY position)
            ), numbered AS (
              SELECT *, sum(CASE WHEN previous_end IS NULL OR previous_end + 1 <> start_line
                                      OR previous_directive <> is_directive
                                      OR previous_documentation <> is_documentation
                                 THEN 1 ELSE 0 END)
                          OVER (PARTITION BY fact_id ORDER BY position) - 1 AS ordinal
              FROM ordered
            ), grouped AS (
              SELECT fact_order, fact_id, path, language, ordinal,
                     any_value(punctuation) AS punctuation,
                     arg_min(start_line, position) AS start_line,
                     arg_min(start_column, position) AS start_column,
                     arg_max(end_line, position) AS end_line,
                     arg_max(end_column, position) AS end_column,
                     sum(line_count) AS line_count, sum(strlen(text)) AS character_count,
                     sum(comment_tokens(text)) AS token_count,
                     string_agg(body, chr(10) ORDER BY position) AS text,
                     any_value(is_directive) AS is_directive,
                     any_value(is_documentation) AS is_documentation
              FROM numbered GROUP BY fact_order, fact_id, path, language, ordinal
            ), spoken AS (
              SELECT g.*, string_agg(
                       decode(array_slice(encode(l.text),
                                          CASE WHEN l.line = g.start_line
                                               THEN g.start_column + 1 ELSE 1 END,
                                          CASE WHEN l.line = g.end_line
                                               THEN g.end_column ELSE strlen(l.text) END)),
                       chr(10) ORDER BY l.line) AS node_text
              FROM grouped g JOIN comment_lines l ON l.fact_id = g.fact_id
                                                 AND l.line BETWEEN g.start_line AND g.end_line
              GROUP BY ALL
            )
            SELECT fact_order, fact_id, ordinal, character_count, is_directive, is_documentation,
                   line_count, fact_id || '/groups:' || ordinal AS record_id,
                   path || ':' || start_line || ':' || start_column || ':comment' AS node_id,
                   'comment' AS node_kind, end_column AS node_span_end_column,
                   end_line AS node_span_end_line, path AS node_span_path,
                   start_column AS node_span_start_column, start_line AS node_span_start_line,
                   node_text,
                   CASE WHEN NOT is_directive AND trim(text) <> ''
                             AND len(list_filter(punctuation,
                                                 lambda p: contains(trim(text), p))) > 0
                        THEN comment_parses(language, trim(text)) ELSE false END
                     AS parses_as_source,
                   text, token_count
            FROM spoken
        """,
    }
