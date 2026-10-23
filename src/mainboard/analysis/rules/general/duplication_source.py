"""Rules over repeated implementation (mcmr's ALL-DUPL source rules), for every language
sitting_duck parses.

A copy is found on normalized tokens: each leaf of a function body's syntax tree in order, a name
standing for any name and a literal for any literal of its kind, so a paste survives renamed
locals and redone formatting. Every window of `_WINDOW` tokens is hashed; equal windows are joined
and grown into maximal runs (one island per pair of functions and offset), so a long paste is one
run rather than one for each shorter reading inside it. A run stays inside one function on each
side, the implementation block mcmr's kernel compared, so module tables, imports and stubs are
never a copy, and a docstring is no token at all: prose, one literal spanning many lines. A window shared by more than `_FANOUT` places is boilerplate rather than a paste and
starts no run.

sitting_duck prunes punctuation, so a run counts fewer tokens than a lexer does: mcmr's default of
60 lexer tokens is about 45 here.
"""

from ...catalog import Example, Relation, Rule

_WINDOW = 20
_FANOUT = 30
_CODE = "f.parses AND f.language NOT IN ('markdown', 'toml') AND NOT f.is_stub"


class CloneRuns(Relation):
    """The `clone_runs()` relation: each repeated run of at least `_WINDOW` normalized tokens,
    located at its later copy (`path`, `start_line`, `end_line`) with the earlier place it repeats
    (`original`, `original_line`, `original_end_line`): how many tokens it runs and on how many of
    the copy's lines they stand."""

    stored = True
    query = f"""
        WITH docstrings AS (
          SELECT d.file_id, s.node_id AS first_id, s.node_id + s.descendant_count AS last_id
          FROM core_def d
          JOIN core_node s ON s.file_id = d.file_id AND s.parent_id = d.body_id
            AND s.semantic_type <> 'METADATA_COMMENT'
          WHERE d.docstring IS NOT NULL
          QUALIFY s.sibling_index = min(s.sibling_index) OVER (PARTITION BY d.file_id, d.def_id)
        ), tokens AS (
          SELECT n.file_id, f.path, n.function_id AS fn, n.start_line AS line, n.end_line,
                 row_number() OVER (PARTITION BY n.file_id ORDER BY n.node_id) AS idx,
                 CASE WHEN n.semantic_type LIKE 'NAME%' THEN 'id'
                      WHEN n.semantic_type LIKE 'LITERAL%' THEN 'lit:' || n.type
                      ELSE n.type END AS token
          FROM core_node n JOIN core_file f USING (file_id)
          LEFT JOIN docstrings d ON d.file_id = n.file_id
            AND n.node_id BETWEEN d.first_id AND d.last_id
          WHERE n.children_count = 0 AND n.function_id IS NOT NULL AND {_CODE}
            AND n.semantic_type NOT LIKE 'METADATA%' AND n.semantic_type NOT LIKE 'PARSER%'
            AND d.file_id IS NULL
        ), grams AS (
          SELECT file_id, idx, fn, hash(string_agg(token, ' ') OVER w) AS h,
                 count(*) OVER w AS width, min(fn) OVER w = max(fn) OVER w AS within
          FROM tokens
          WINDOW w AS (PARTITION BY file_id ORDER BY idx
                       ROWS BETWEEN CURRENT ROW AND {_WINDOW - 1} FOLLOWING)
        ), usable AS (
          SELECT * FROM grams WHERE width = {_WINDOW} AND within
          QUALIFY count(*) OVER (PARTITION BY h) BETWEEN 2 AND {_FANOUT}
        ), pairs AS (
          SELECT a.file_id AS fa, a.fn AS fna, a.idx AS ia, b.file_id AS fb, b.fn AS fnb,
                 b.idx AS ib
          FROM usable a JOIN usable b ON a.h = b.h
           AND (a.file_id < b.file_id OR (a.file_id = b.file_id AND a.idx < b.idx))
        ), runs AS (
          SELECT fa, fb, min(ia) AS ia, min(ib) AS ib, count(*) + {_WINDOW - 1} AS tokens
          FROM (SELECT *, ia - row_number() OVER (PARTITION BY fa, fna, fb, fnb, ib - ia
                                                  ORDER BY ia) AS island
                FROM pairs)
          GROUP BY fa, fna, fb, fnb, ib - ia, island
          HAVING fa <> fb OR ib - ia >= count(*) + {_WINDOW - 1}
        )
        SELECT b.path, b.line AS start_line, be.end_line, a.path AS original,
               a.line AS original_line, ae.end_line AS original_end_line, r.tokens,
               (SELECT count(DISTINCT t.line) FROM tokens t
                WHERE t.file_id = r.fb AND t.idx BETWEEN r.ib AND r.ib + r.tokens - 1) AS lines
        FROM runs r
        JOIN tokens a ON a.file_id = r.fa AND a.idx = r.ia
        JOIN tokens ae ON ae.file_id = r.fa AND ae.idx = r.ia + r.tokens - 1
        JOIN tokens b ON b.file_id = r.fb AND b.idx = r.ib
        JOIN tokens be ON be.file_id = r.fb AND be.idx = r.ib + r.tokens - 1
    """


class CodeLines(Relation):
    """The `code_lines()` relation: how many lines the code files that clones are read from hold,
    the denominator of a duplicated share."""

    query = f"""
        SELECT coalesce(sum(lines), 0)::BIGINT AS lines
        FROM (SELECT n.file_id, max(n.end_line) AS lines
              FROM core_node n JOIN core_file f USING (file_id) WHERE {_CODE} GROUP BY ALL)
    """


class PastedBlockCopy(Rule, id="ALL-DUPL0003"):
    """Copies of an implementation block, matched on normalized tokens so renamed locals and redone
    formatting still read as the same paste: one finding per copy past the first, at the copy,
    naming the original. Whether the copies share knowledge is a judgment this rule leaves to
    review; raising `minimum_tokens` is the honest way to ask for fewer findings."""

    minimum_tokens: int = 45
    minimum_lines: int = 4

    query = """
        SELECT path, start_line, 0 AS start_column, end_line, original, original_line, tokens,
               lines
        FROM clone_runs()
        WHERE tokens >= $minimum_tokens AND lines >= $minimum_lines
        QUALIFY row_number() OVER (PARTITION BY path, start_line
                                   ORDER BY original, original_line) = 1
    """
    message = (
        "these {lines} lines ({tokens} tokens) repeat `{original}` from line {original_line}; "
        "extract the shared block or record why both exist"
    )
    examples = (
        Example(
            bad=(
                "def total_over(rows, limit):\n    kept = []\n    for row in rows:\n"
                "        if row.value > limit and row.active and not row.hidden:\n"
                "            kept.append(row.value * row.weight + row.offset)\n"
                "    kept.sort(key=abs, reverse=True)\n"
                "    return sum(kept) / max(len(kept), 1)\n\n\n"
                "def mean_above(items, floor):\n    chosen = []\n    for item in items:\n"
                "        if item.value > floor and item.active and not item.hidden:\n"
                "            chosen.append(item.value * item.weight + item.offset)\n"
                "    chosen.sort(key=abs, reverse=True)\n"
                "    return sum(chosen) / max(len(chosen), 1)\n"
            ),
            good=(
                "def total_over(rows, limit):\n    kept = []\n    for row in rows:\n"
                "        if row.value > limit and row.active and not row.hidden:\n"
                "            kept.append(row.value * row.weight + row.offset)\n"
                "    kept.sort(key=abs, reverse=True)\n"
                "    return sum(kept) / max(len(kept), 1)\n\n\n"
                "def mean_above(items, floor):\n    return total_over(items, floor)\n"
            ),
        ),
    )


class DuplicatedRepositoryShare(Rule, id="ALL-DUPL0004"):
    """The share of the repository's code lines one repeated block accounts for: the lines its
    copies past the first repeat over every code line read, as a percentage, reported per block
    so the fix is one particular extraction rather than duplication in general."""

    maximum_percent: float = 3.0
    minimum_lines: int = 4

    query = """
        WITH blocks AS (
          SELECT original AS path, original_line AS start_line, count(*) AS copies,
                 sum(lines) AS repeated
          FROM (SELECT DISTINCT path, start_line, lines, original, original_line
                FROM clone_runs() WHERE lines >= $minimum_lines)
          GROUP BY original, original_line
        )
        SELECT b.path, b.start_line, 0 AS start_column, b.copies, b.repeated, c.lines AS total,
               round(100.0 * b.repeated / c.lines, 2) AS value
        FROM blocks b, code_lines() c
        WHERE c.lines > 0 AND 100.0 * b.repeated / c.lines > $maximum_percent
    """
    message = "this block's {copies} copies repeat {repeated} of the {total} code lines ({value}%)"


class RepeatedSiblingMethod(Rule, id="ALL-DUPL0001", applicability="review"):
    """Methods written identically in sibling classes of one project base, a concrete pull-up
    boundary: decorators, name, signature and statements match node for node once the docstring
    and comments are set aside. A placeholder body (empty, `pass`, `...`,
    `raise NotImplementedError`, `return NotImplemented`, as its leaves read joined by spaces) is
    not an implementation, and classes sharing no resolved project base are never compared, since
    no owner for the extraction can be proven. When the base already defines the method its own
    way, the copies belong in an intermediate class rather than the base.

    The fix pulls the first copy up and deletes them all, only where nothing can resolve
    differently: the base lacks the method, the base and every copy are module classes of one
    file (the body reads the same names there), and each sibling keeps another statement."""

    placeholder: str = r"^(pass|\.\.\.|raise NotImplementedError( \( \))?|return NotImplemented)?$"

    query = r"""
        WITH defs AS (
          SELECT DISTINCT m.method_id, m.class_id, m.name, m.path, m.start_line, m.start_column,
                 d.file_id, coalesce(d.wrapper_id, d.def_id) AS first_id,
                 d.def_id + d.size AS last_id, d.body_id, d.docstring IS NOT NULL AS documented,
                 d.start_line AS first_line, d.end_line AS last_line,
                 c.scope = 'module' AND k.statements >= 2 AS movable
          FROM class_methods m
          JOIN class_classes c ON c.class_id = m.class_id AND NOT c.is_test
          JOIN core_file f ON f.path = m.path
          JOIN core_def d ON d.file_id = f.file_id AND d.kind = 'function' AND d.name = m.name
            AND d.end_line = m.end_line AND d.end_column = m.end_column
          JOIN core_def k ON k.file_id = d.file_id AND k.def_id = d.class_id
        ), docstrings AS (
          SELECT d.method_id, s.node_id AS first_id, s.node_id + s.descendant_count AS last_id
          FROM defs d
          JOIN core_node s ON s.file_id = d.file_id AND s.parent_id = d.body_id
            AND s.semantic_type <> 'METADATA_COMMENT'
          WHERE d.documented
          QUALIFY s.sibling_index = min(s.sibling_index) OVER (PARTITION BY d.method_id)
        ), shapes AS (
          SELECT d.method_id,
                 string_agg(n.type || '=' || n.name, ' ' ORDER BY n.node_id) AS shape,
                 coalesce(string_agg(coalesce(nullif(n.name, ''), n.type), ' ' ORDER BY n.node_id)
                            FILTER (WHERE n.node_id > d.body_id AND n.children_count = 0), '')
                   AS body
          FROM defs d
          LEFT JOIN docstrings s USING (method_id)
          JOIN core_node n ON n.file_id = d.file_id
            AND n.node_id BETWEEN d.first_id AND d.last_id
            AND n.node_id NOT BETWEEN coalesce(s.first_id, -1) AND coalesce(s.last_id, -1)
            AND n.semantic_type <> 'METADATA_COMMENT'
          GROUP BY d.method_id
        ), grouped AS (
          SELECT d.*, e.parent AS base,
                 count(DISTINCT d.class_id) OVER copies AS siblings,
                 row_number() OVER copies AS rank,
                 list({'path': d.path, 'first_line': d.first_line, 'last_line': d.last_line,
                       'movable': d.movable}) OVER (copies ROWS BETWEEN UNBOUNDED PRECEDING
                                                   AND UNBOUNDED FOLLOWING) AS members
          FROM defs d
          JOIN shapes s USING (method_id)
          JOIN override_edges e ON e.child = d.class_id
          WHERE NOT regexp_matches(s.body, $placeholder)
          WINDOW copies AS (PARTITION BY e.parent, s.shape ORDER BY d.path, d.start_line)
        )
        SELECT *, CASE WHEN defined
                       THEN 'share it through an intermediate class, since `' || base
                            || '` defines its own'
                       ELSE 'pull it up into `' || base || '`' END AS remedy
        FROM (SELECT DISTINCT ON (g.method_id, g.base)
                     g.path, g.start_line, g.start_column, g.name, g.rank, g.members,
                     g.members[1].path || ':' || g.members[1].first_line AS first,
                     b.name AS base, b.path AS base_path, b.end_line AS base_end,
                     b.scope = 'module' AS base_movable,
                     EXISTS (SELECT 1 FROM class_methods x
                             WHERE x.class_id = g.base AND x.name = g.name) AS defined
              FROM grouped g JOIN class_classes b ON b.class_id = g.base
              WHERE g.siblings >= 2 AND g.rank > 1)
    """
    message = "`{name}` repeats `{first}` in another subclass of `{base}`; {remedy}"
    fix = """
        WITH pulled AS (
          SELECT f.finding_key, f.base_path AS path, f.base_end, f.members,
                 string_split(s.content, chr(10)) AS lines
          FROM finding f JOIN source_files s ON s.path = f.base_path
          WHERE f.rank = 2 AND NOT f.defined AND f.base_movable
            AND list_bool_and(list_transform(f.members,
                  lambda m: m.path = f.base_path AND m.movable))
        )
        SELECT finding_key, path, base_end AS start_line,
               strlen(rtrim(lines[base_end], chr(13))) AS start_column, base_end AS end_line,
               strlen(rtrim(lines[base_end], chr(13))) AS end_column,
               chr(10) || chr(10) || array_to_string(
                 lines[members[1].first_line:members[1].last_line], chr(10)) AS text
        FROM pulled
        UNION ALL BY NAME
        SELECT finding_key, path, functions_member_start(lines, m.first_line) AS start_line,
               0 AS start_column, m.last_line + 1 AS end_line, 0 AS end_column, '' AS text
        FROM (SELECT finding_key, path, lines, unnest(members) AS m FROM pulled)
    """
    examples = (
        Example(
            bad=(
                "class Keyed:\n    name: str\n\n\nclass Category(Keyed):\n"
                "    kind = 'category'\n\n    @property\n    def key(self):\n"
                '        """The category\'s key."""\n'
                "        return self.name.lower().replace(' ', '-')\n\n\n"
                "class Criterion(Keyed):\n    kind = 'criterion'\n\n    @property\n"
                "    def key(self):\n        # the criterion's key\n"
                "        return self.name.lower().replace(' ', '-')\n"
            ),
            good=(
                "class Keyed:\n    name: str\n\n    @property\n    def key(self):\n"
                '        """The category\'s key."""\n'
                "        return self.name.lower().replace(' ', '-')\n\n\n"
                "class Category(Keyed):\n    kind = 'category'\n\n\n"
                "class Criterion(Keyed):\n    kind = 'criterion'\n"
            ),
        ),
    )
