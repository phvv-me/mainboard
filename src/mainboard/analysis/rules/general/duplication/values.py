"""Rules over repeated string literals (mcmr's ALL-DUPL value rules)."""

from ....catalog import Example, Relation, Rule


class StringGroups(Relation):
    """The string expressions of each Python file grouped by what they say and the role they
    play: `value` (anywhere else, including the called expression itself), `argument` (a
    positional argument), `keyword` (a keyword argument or `**` spread, and what it holds but a
    dict), `entry` (a key or a value of a dict literal a keyword holds) or `comparison` (either
    side of a single `==`), with how many times the file writes the group (`occurrence_count`)
    and where it first does (`first_line`, `first_column`). `is_excluded_vocabulary` marks the
    roles ALL-DUPL0002 leaves out (`value`, `argument`, `entry`) and `is_callee_vocabulary` the
    ones that hand the text to a callable, which ALL-DUPL0005 leaves out (`argument`, `keyword`,
    `entry`). A group lives in one file and may be written once there, and a text of three
    characters or fewer makes none. The keys of a dict literal in a test module are not counted:
    an expectation spells its record's fields."""

    stored = True
    query = r"""
        WITH python AS (
          SELECT n.file_id, n.node_id AS id, n.parent_id AS pid, n.sibling_index AS sib, n.type,
                 n.children_count AS cc, n.descendant_count AS dc
          FROM core_node n JOIN core_file f USING (file_id)
          WHERE f.language = 'python'
            AND n.type IN ('call', 'argument_list', 'generator_expression', 'keyword_argument',
                           'dictionary_splat', 'comparison_operator', '==', 'dictionary')
        ), strings AS (
          SELECT file_id, text_id AS id, path, value, start_line, start_column FROM core_text
          WHERE kind = 'string' AND plain AND repeat_id IS NULL
            AND parent_type IS DISTINCT FROM 'expression_statement'
            AND NOT (parent_type = 'pair' AND import_is_test_path(path) AND EXISTS (
                       SELECT 1 FROM core_node k
                       WHERE k.file_id = core_text.file_id AND k.node_id = core_text.text_id
                         AND k.sibling_index = 0))
          UNION ALL
          SELECT file_id, repeat_id, path, value, start_line, start_column FROM core_text
          WHERE kind = 'string' AND repeat_count IS NOT NULL
        ), setters AS (
          SELECT file_id, id, id + dc AS last, 'value' AS role FROM python WHERE type = 'call'
          UNION ALL
          SELECT a.file_id, a.id, a.id + a.dc, 'argument'
          FROM python a JOIN python c ON c.file_id = a.file_id AND c.id = a.pid
            AND c.type = 'call'
          WHERE a.type IN ('argument_list', 'generator_expression')
          UNION ALL
          SELECT k.file_id, k.id, k.id + k.dc, 'keyword'
          FROM python k
          JOIN python a ON a.file_id = k.file_id AND a.id = k.pid AND a.type = 'argument_list'
          JOIN python c ON c.file_id = a.file_id AND c.id = a.pid AND c.type = 'call'
          WHERE k.type IN ('keyword_argument', 'dictionary_splat')
          UNION ALL
          SELECT c.file_id, c.id, c.id + c.dc, 'comparison'
          FROM python c JOIN python o ON o.file_id = c.file_id AND o.pid = c.id AND o.sib = 1
            AND o.type = '=='
          WHERE c.type = 'comparison_operator' AND c.cc = 3
          UNION ALL
          SELECT file_id, id, id + dc, 'entry' FROM python WHERE type = 'dictionary'
        ), enclosed AS (
          SELECT s.path, s.value, s.start_line, s.start_column,
                 coalesce(arg_max(t.role, t.id) FILTER (WHERE t.role <> 'entry'), 'value') AS role,
                 coalesce(max(t.id) FILTER (WHERE t.role <> 'entry'), -1) AS role_id,
                 coalesce(max(t.id) FILTER (WHERE t.role = 'entry'), -1) AS dictionary_id
          FROM strings s LEFT JOIN setters t ON t.file_id = s.file_id AND t.id < s.id
            AND s.id <= t.last
          GROUP BY s.file_id, s.id, s.path, s.value, s.start_line, s.start_column
        ), roles AS (
          SELECT path, value, start_line, start_column,
                 CASE WHEN role = 'keyword' AND dictionary_id > role_id THEN 'entry'
                      ELSE role END AS role
          FROM enclosed
        ), grouped AS (
          SELECT path, value, role, count(*)::BIGINT AS occurrence_count,
                 min({'line': start_line, 'col': start_column}) AS earliest
          FROM roles GROUP BY path, value, role HAVING strlen(value) > 3
        )
        SELECT path, 'literalgroupfact:' || path AS fact_id, value, role, occurrence_count,
               earliest.line AS first_line, earliest.col AS first_column,
               role IN ('value', 'argument', 'entry') AS is_excluded_vocabulary,
               role IN ('argument', 'keyword', 'entry') AS is_callee_vocabulary
        FROM grouped
    """


class RepeatedSemanticStringLiteral(Rule, id="ALL-DUPL0002"):
    """Repeated string literals that occupy one exact semantic role across several files.

    A group is one text in one role: the value of a keyword argument (a literal tuple or list it
    holds included) or one side of an `==`. Its occurrences count over every file, so a text each
    of three files passes once is as repeated as one a single file writes three times; it is
    reported when they reach `minimum_occurrences` in at least `minimum_files` files, and each
    file that holds it gets the finding at the text's first line there.

    A dict literal's keys and values are not keyword values: a key names a field, and the value
    takes its role from that key, so `headers={"Content-Type": ...}` and `media_type=...` are two
    roles.
    """

    minimum_occurrences: int = 3
    minimum_files: int = 2
    minimum_length: int = 12

    query = r"""
        WITH shared AS (
          SELECT value, role, sum(occurrence_count) AS occurrences, count(*) AS files
          FROM string_groups()
          WHERE NOT is_excluded_vocabulary AND length(value) >= $minimum_length
          GROUP BY value, role
          HAVING sum(occurrence_count) >= $minimum_occurrences AND count(*) >= $minimum_files
        )
        SELECT g.path, g.first_line AS start_line, g.first_column AS start_column,
               left(replace(g.value, chr(10), '\n'), 80) AS literal, g.role,
               s.occurrences, s.files
        FROM string_groups() g JOIN shared s ON s.value = g.value AND s.role = g.role
    """
    message = "`{literal}` fills the {role} role {occurrences} times across {files} files"
    examples = (
        Example(
            bad=(
                """def publish(client):
    client.send(topic="audit-events-stream")
    client.send(topic="audit-events-stream")
"""
            ),
            good=(
                """from .topics import AUDIT_EVENTS


def publish(client):
    client.send(topic=AUDIT_EVENTS)
    client.send(topic=AUDIT_EVENTS)
"""
            ),
            path="ledger/first.py",
            context={
                "ledger/__init__.py": "",
                "ledger/second.py": 'def replay(client):\n    client.send(topic="audit-events-stream")\n',
            },
        ),
        Example(
            bad='def fast(mode):\n    return mode == "fast-path-literal"\n',
            good=(
                """from .modes import FAST_PATH


def fast(mode):
    return mode == FAST_PATH
"""
            ),
            path="ledger/first.py",
            context={
                "ledger/__init__.py": "",
                "ledger/second.py": 'def quick(mode):\n    return mode == "fast-path-literal"\n',
                "ledger/third.py": 'def rapid(mode):\n    return mode == "fast-path-literal"\n',
            },
        ),
        Example(
            bad='def publish(client):\n    client.send(topic="audit-events-stream")\n',
            good=(
                """from .topics import AUDIT_EVENTS


def publish(client):
    client.send(topic=AUDIT_EVENTS)
"""
            ),
            path="ledger/first.py",
            context={
                "ledger/__init__.py": "",
                "ledger/second.py": """def replay(client):
    client.send(topic="audit-events-stream")
    client.send(topic="audit-events-stream")
""",
                "ledger/third.py": 'def tag(client):\n    client.send(headers={"audit-events-stream": 1})\n',
            },
        ),
    )


class ModuleRepeatedStringLiteral(Rule, id="ALL-DUPL0005", applicability="review"):
    """One string literal a single module writes out over and over.

    The finding sits at the first place the module writes it, the text it hands to a callable
    left out.
    """

    minimum_occurrences: int = 4
    minimum_length: int = 8

    query = """
        SELECT path, earliest.line AS start_line, earliest.col AS start_column, fact_id, literal,
               module_occurrences
        FROM (
          SELECT path, fact_id, value AS literal, sum(occurrence_count) AS module_occurrences,
                 min({'line': first_line, 'col': first_column}) AS earliest
          FROM string_groups() WHERE NOT is_callee_vocabulary AND occurrence_count > 1
          GROUP BY path, fact_id, value
          HAVING length(value) >= $minimum_length AND sum(occurrence_count) >= $minimum_occurrences)
    """
    message = "`{literal}` is written {module_occurrences} times in this module"
    # Python only, for a value of word, path and flag characters (`observed`, `staged.txt`,
    # `/usr/bin/xcrun`, `--version`, `runtimeWarnings`), its name its words in capitals
    # (`_STAGED_TXT`, `_RUNTIME_WARNINGS`). A capitalized word, or a dotted name ending in one,
    # stays out: it is most often a type named in a string annotation, where a constant is no
    # longer the type. A module constant `_OBSERVED = "observed"` goes
    # after the last top-level import and replaces every occurrence, all of a file's values in one
    # group so that its findings clear together. A value is left alone when any of its
    # occurrences stands alone on its line (a docstring), spans lines, comes before the
    # constant's place, when fewer occurrences are found than the finding counts (assignment
    # targets are not among the string expressions), or when the module uses `Literal[` or
    # `case` (where a name is no longer the string), already defines the value as a symbol, or
    # the constant's name is taken.
    fix = r"""
        WITH candidates AS (
          SELECT f.finding_key, f.path, f.literal, f.module_occurrences,
                 '_' || upper(trim(regexp_replace(
                   regexp_replace(f.literal, '([a-z0-9])([A-Z])', '\1_\2', 'g'),
                   '[^A-Za-z0-9]+', '_', 'g'), '_')) AS name
          FROM finding f
          WHERE regexp_matches(f.literal, '^[A-Za-z0-9_./ -]{1,60}$')
            AND regexp_matches(f.literal, '[A-Za-z]')
            AND NOT regexp_matches(f.literal, '^([A-Za-z_]\w*\.)*[A-Z]\w*$')
            AND regexp_matches(f.path, '\.pyi?$')
        ), occurrences AS (
          SELECT DISTINCT c.finding_key, c.path, c.literal, c.name,
                 e.start_line, e.start_column, e.end_line, e.end_column,
                 trimmed(l.text) = e.text AS alone
          FROM candidates c
          JOIN core_text e
            ON e.path = c.path AND e.kind = 'string' AND e.plain AND e.repeat_id IS NULL
           AND e.parent_type IS DISTINCT FROM 'expression_statement' AND e.value = c.literal
          JOIN source_lines() l ON l.path = c.path AND l.line = e.start_line
        ), places AS (
          SELECT path, max(end_line) AS after_imports FROM import_binding_nodes
          WHERE role = 'declaration' AND start_column = 0 GROUP BY path
        ), usable AS (
          SELECT c.*, p.after_imports FROM candidates c
          JOIN places p ON p.path = c.path
          JOIN source_files s ON s.path = c.path
          WHERE NOT contains(s.content, 'Literal[')
            AND NOT regexp_matches(s.content, '(?m)^[ \t]*case[ \t]')
            AND NOT regexp_matches(s.content, '\b' || c.name || '\b')
            AND NOT EXISTS (SELECT 1 FROM control_artifacts_module_names() y
                            WHERE y.path = c.path AND y.name IN (c.literal, c.name))
            AND NOT EXISTS (SELECT 1 FROM occurrences o
                            WHERE o.finding_key = c.finding_key
                              AND (o.alone OR o.start_line <> o.end_line
                                   OR o.start_line <= p.after_imports))
            AND c.module_occurrences <= (SELECT count(*) FROM occurrences o
                                        WHERE o.finding_key = c.finding_key)
          QUALIFY count(*) OVER (PARTITION BY c.path, c.name) = 1
        ), grouped AS (
          SELECT path, min(finding_key) AS finding_key, max(after_imports) AS after_imports,
                 string_agg(name || ' = "' || literal || '"', chr(10) ORDER BY finding_key)
                   AS constants
          FROM usable GROUP BY path
        )
        SELECT g.finding_key, g.path, g.after_imports + 1 AS start_line, 0 AS start_column,
               g.after_imports + 1 AS end_line, 0 AS end_column,
               chr(10) || g.constants || chr(10) AS text
        FROM grouped g
        UNION ALL BY NAME
        SELECT g.finding_key, o.path, o.start_line, o.start_column, o.end_line, o.end_column,
               o.name AS text
        FROM occurrences o JOIN usable u USING (finding_key) JOIN grouped g ON g.path = u.path
    """
    examples = (
        Example(
            bad=(
                """import sys

SHAPES = [
    ("q_proj", "self_attn", 2048),
    ("k_proj", "self_attn", 1024),
    ("v_proj", "self_attn", 1024),
    ("o_proj", "self_attn", 1024),
]
"""
            ),
            good=(
                """import sys

_SELF_ATTN = "self_attn"

SHAPES = [
    ("q_proj", _SELF_ATTN, 2048),
    ("k_proj", _SELF_ATTN, 1024),
    ("v_proj", _SELF_ATTN, 1024),
    ("o_proj", _SELF_ATTN, 1024),
]
"""
            ),
        ),
        Example(
            bad=(
                """import subprocess

TOOLS = {
    "ruff": ["ruff", "--version"],
    "ty": ["ty", "--version"],
    "uv": ["uv", "--version"],
    "pixi": ["pixi", "--version"],
}
"""
            ),
            good=(
                """import subprocess

_VERSION = "--version"

TOOLS = {
    "ruff": ["ruff", _VERSION],
    "ty": ["ty", _VERSION],
    "uv": ["uv", _VERSION],
    "pixi": ["pixi", _VERSION],
}
"""
            ),
        ),
    )
