"""mcmr's RouteFact, derived from the text of the project's source files.

mcmr reads routes lexically, one small reader per language, and so does this derivation (a syntax
tree would add routes mcmr does not see, such as a decorator whose path sits on the next line):

- Python: a decorator line `@x.<verb>("/path"...)`;
- TypeScript and JavaScript: a line `x.<verb>("/path", handler)` (the comma tells a registration
  from a client call);
- Rust: an attribute `#[<verb>("/path")]`, or a builder `.route("/path", get(a).post(b))`, one
  route per verb the call names;
- a directory convention, whatever the language: SvelteKit `routes/.../+server.ts` and the Next
  app router `app/.../route.ts` serve the path their directories spell.

Verbs are `get post put patch delete head options route websocket ws`. A line is read only when it
is code: lines in a triple-quoted block, comment lines and everything after `#[cfg(test)]` are
dropped, and so are files that illustrate rather than serve (under `tests/`, `test_*`, `*.test.*`,
`*.spec.*`). A route is prefix-composed when its file mentions a router prefix (`prefix=`,
`.nest(`, `app.use(`, `include_router(`, `.scope(`, `url_prefix=`). A reference to a route is
another Python, Rust, TypeScript, JavaScript, Svelte or Go file naming its path in quotes.

Differences from the kernel, kept on purpose: the facts do not state `frameworks` and each route's
`references` list (only their count), which no rule reads.
"""

from typing import TYPE_CHECKING

from ...catalog import Family, Function
from ...engine.facts import listed

_ROUTE_CORPUS_TABLE_SQL = "CREATE TABLE route_corpus (path VARCHAR)"

if TYPE_CHECKING:
    from pathlib import Path

    from ...engine.facts import Facts

_SUFFIXES = ("py", "rs", "ts", "tsx", "js", "jsx", "mjs", "svelte", "go")
_VERBS = "['get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'route', 'websocket', 'ws']"


class RouteLiteral(Function):
    """The first quoted string of an argument list when it is a path (starts with `/`, holds no
    space), NULL otherwise."""

    arguments: str

    expression = r"""
        list_transform(
          [nullif(least(coalesce(nullif(instr(arguments, '"'), 0), 1073741824),
                        coalesce(nullif(instr(arguments, chr(39)), 0), 1073741824),
                        coalesce(nullif(instr(arguments, '`'), 0), 1073741824)), 1073741824)],
          lambda opening: list_transform(
            [substr(arguments, opening + 1)],
            lambda rest: list_transform(
              [substr(rest, 1, instr(rest, substr(arguments, opening, 1)) - 1)],
              lambda stated: CASE WHEN instr(rest, substr(arguments, opening, 1)) > 0
                                       AND starts_with(stated, '/') AND NOT contains(stated, ' ')
                                  THEN stated END)[1])[1])[1]
    """


class RouteStrip(Function):
    """A line without leading and trailing whitespace."""

    line: str

    expression = r"regexp_replace(line, '^\s+|\s+$', '', 'g')"


class Route(Family):
    family = "RouteFact"
    relations = {
        "facts": r"""
            SELECT 0 AS fact_order, 'routes' AS fact_id, '' AS path, 1 AS start_line,
                   0 AS start_column, 1 AS end_line, 0 AS end_column
        """,
        "files": r"""
            SELECT c.path, s.content,
                   CASE regexp_extract(c.path, '[^.]*$')
                     WHEN 'py' THEN 'python' WHEN 'rs' THEN 'rust' WHEN 'go' THEN 'other'
                     ELSE 'typescript' END AS kind,
                   (starts_with(c.path, 'tests/') OR contains(c.path, '/tests/')
                    OR starts_with(regexp_extract(c.path, '[^/]*$'), 'test_')
                    OR contains(regexp_extract(c.path, '[^/]*$'), '.test.')
                    OR contains(regexp_extract(c.path, '[^/]*$'), '.spec.')) AS is_example,
                   len(list_filter(
                     ['prefix=', '.nest(', 'app.use(', 'include_router(', '.scope(',
                      'url_prefix='],
                     lambda marker: contains(s.content, marker))) > 0
                     AS is_prefix_composed
            FROM route_corpus c JOIN source_files s USING (path)
        """,
        "lines": r"""
            WITH bodies AS (
              SELECT path, kind, split_part(content, '#[cfg(test)]', 1) AS body,
                     contains(content, chr(34) || chr(34) || chr(34))
                       OR contains(content, chr(39) || chr(39) || chr(39)) AS has_blocks
              FROM route_files WHERE NOT is_example AND kind <> 'other'
            ), lined AS (
              SELECT path, kind, has_blocks, generate_subscripts(a.lines, 1) AS line,
                     unnest(a.lines) AS raw
              FROM (SELECT path, kind, has_blocks, string_split(body, chr(10)) AS lines
                    FROM bodies) a
            ), stripped AS (
              SELECT *, regexp_replace(raw, '\r$', '') AS text,
                     route_strip(regexp_replace(raw, '\r$', '')) AS trimmed
              FROM lined
            ), counted AS (
              SELECT *, (len(regexp_extract_all(trimmed, chr(34) || chr(34) || chr(34)))
                         + len(regexp_extract_all(trimmed, chr(39) || chr(39) || chr(39)))) % 2
                          AS odd,
                     (starts_with(trimmed, '#') AND NOT starts_with(trimmed, '#['))
                       OR starts_with(trimmed, '//') OR starts_with(trimmed, '*')
                       OR starts_with(trimmed, '/*') AS commented
              FROM stripped
            ), blocked AS (
              SELECT *, coalesce(sum(odd) OVER (PARTITION BY path ORDER BY line
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) % 2
                          AS opened
              FROM counted WHERE has_blocks
              UNION ALL BY NAME
              SELECT *, 0 AS opened FROM counted WHERE NOT has_blocks
            )
            SELECT path, kind, line, text, trimmed
            FROM blocked
            WHERE opened = 0 AND (opened + odd) % 2 = 0 AND NOT commented
              AND CASE kind WHEN 'python' THEN starts_with(trimmed, '@')
                            WHEN 'rust' THEN starts_with(trimmed, '#[')
                                             OR contains(trimmed, '.route(')
                            ELSE contains(text, '(')
                                 AND contains(substr(text, instr(text, '(') + 1), ',') END
        """,
        "routes": rf"""
            WITH verbs AS (
              SELECT unnest({_VERBS}) AS verb, generate_subscripts({_VERBS}, 1) AS verb_order
            ), decorated AS (
              SELECT path, line, verb, 'decorator' AS framework, 0 AS verb_order,
                     route_literal(arguments) AS route
              FROM (SELECT path, line, substr(called, 1, instr(called, '(') - 1) AS verb,
                           substr(called, instr(called, '(') + 1) AS arguments
                    FROM (SELECT path, line, substr(trimmed, instr(trimmed, '.') + 1) AS called
                          FROM route_lines
                          WHERE kind = 'python' AND instr(trimmed, '.') > 0
                            AND starts_with(trimmed, '@'))
                    WHERE instr(called, '(') > 0)
            ), registered AS (
              SELECT path, line, route_strip(regexp_extract(before, '[^.]*$')) AS verb,
                     'registration' AS framework, 0 AS verb_order,
                     route_literal(arguments) AS route
              FROM (SELECT path, line, substr(text, 1, instr(text, '(') - 1) AS before,
                           substr(text, instr(text, '(') + 1) AS arguments
                    FROM route_lines WHERE kind = 'typescript')
              WHERE contains(before, '.') AND contains(arguments, ',')
            ), attributes AS (
              SELECT path, line, substr(rest, 1, instr(rest, '(') - 1) AS verb,
                     'attribute' AS framework, 0 AS verb_order,
                     route_literal(substr(rest, instr(rest, '(') + 1)) AS route
              FROM (SELECT path, line, substr(trimmed, 3) AS rest FROM route_lines
                    WHERE kind = 'rust' AND starts_with(trimmed, '#['))
              WHERE instr(rest, '(') > 0
            ), builders AS (
              SELECT path, line, verb, 'builder' AS framework, verb_order, route
              FROM (SELECT path, line, route_literal(arguments) AS route, arguments
                    FROM (SELECT path, line,
                                 substr(trimmed, instr(trimmed, '.route(') + 7) AS arguments
                          FROM route_lines WHERE kind = 'rust' AND contains(trimmed, '.route(')))
              JOIN verbs ON contains(arguments, verb || '(')
              WHERE route IS NOT NULL
            ), written AS (
              SELECT path, line, verb, framework, verb_order, route FROM decorated
              WHERE list_contains({_VERBS}, verb) AND route IS NOT NULL
              UNION ALL
              SELECT path, line, verb, framework, verb_order, route FROM registered
              WHERE list_contains({_VERBS}, verb) AND route IS NOT NULL
              UNION ALL
              SELECT path, line, verb, framework, verb_order, route FROM attributes
              WHERE list_contains({_VERBS}, verb) AND route IS NOT NULL
              UNION ALL
              SELECT * FROM builders b
              WHERE NOT EXISTS (SELECT 1 FROM attributes a
                                WHERE a.path = b.path AND a.line = b.line AND a.route IS NOT NULL
                                  AND list_contains({_VERBS}, a.verb))
            ), conventional AS (
              SELECT path, 1 AS line, 'any' AS verb, 'convention' AS framework, 0 AS verb_order,
                     '/' || array_to_string(list_transform(
                       list_filter(list_slice(string_split(inside, '/'), 1,
                                              list_position(string_split(inside, '/'), name) - 1),
                                   lambda step: NOT starts_with(step, '(')),
                       lambda step: CASE WHEN starts_with(step, '[')
                                         THEN '{{' || rtrim(substr(step, 2), ']') || '}}'
                                         ELSE step END), '/') AS route
              FROM (SELECT path, name, anchor, substr(path, instr(path, anchor || '/')
                                                            + length(anchor) + 1) AS inside
                    FROM (SELECT path, regexp_extract(path, '[^/]*$') AS name,
                                 CASE WHEN regexp_extract(path, '[^/]*$') IN
                                        ('+server.ts', '+server.js', '+page.svelte', '+page.ts')
                                        THEN 'routes'
                                      WHEN regexp_extract(path, '[^/]*$') IN
                                        ('route.ts', 'route.js', 'page.tsx', 'page.ts')
                                        THEN 'app' END AS anchor
                          FROM route_files WHERE NOT is_example)
                    WHERE anchor IS NOT NULL AND contains(path, anchor || '/'))
            ), found AS (
              SELECT w.route AS path, w.verb AS method, w.framework, w.path AS declared_in,
                     w.line, w.verb_order, f.is_prefix_composed
              FROM (SELECT * FROM written UNION ALL SELECT * FROM conventional) w
              JOIN route_files f ON f.path = w.path
            ), quoted AS (
              SELECT path AS file,
                     list_distinct(list_transform(
                       regexp_extract_all(content, '["''`]/[^"''`\s]*["''`]'),
                       lambda token: substr(token, 2, length(token) - 2))) AS tokens
              FROM route_files WHERE (SELECT count(*) FROM found) > 0
            ), mentioned AS (
              SELECT f.path, f.declared_in, count(DISTINCT q.file) AS references_length
              FROM (SELECT DISTINCT path, declared_in FROM found) f
              JOIN (SELECT file, unnest(tokens) AS token FROM quoted) q ON q.token = f.path
              JOIN route_files s ON s.path = q.file AND s.path <> f.declared_in
              WHERE contains(s.content, '"' || f.path || '"')
                 OR contains(s.content, chr(39) || f.path || chr(39))
                 OR contains(s.content, '`' || f.path || '`')
              GROUP BY f.path, f.declared_in
            )
            SELECT 0 AS fact_order, 'routes' AS fact_id,
                   'routes/routes:' || (ordinal - 1) AS record_id, ordinal - 1 AS ordinal,
                   method, path, framework, declared_in, line, is_prefix_composed,
                   coalesce(references_length, 0) AS references_length
            FROM (SELECT f.*, m.references_length,
                         row_number() OVER (ORDER BY f.path, f.method, f.declared_in, f.line,
                                                     f.verb_order) AS ordinal
                  FROM found f LEFT JOIN mentioned m
                    ON m.path = f.path AND m.declared_in = f.declared_in)
        """,
    }

    @classmethod
    def provide(cls, facts: Facts, root: Path) -> None:
        """List the files routes are read from and mentioned in, as `route_corpus`.

        `source_files` then reads them.
        """
        paths = [path for path in listed(root) if path.rsplit(".", 1)[-1] in _SUFFIXES]
        facts.db.execute(_ROUTE_CORPUS_TABLE_SQL)
        if paths:
            facts.db.executemany("INSERT INTO route_corpus VALUES (?)", [(p,) for p in paths])
