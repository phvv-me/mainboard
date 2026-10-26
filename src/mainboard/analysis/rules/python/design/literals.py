"""Rules over literals written out inside functions that read better as module constants."""

from ....catalog import Example, Relation, Rule


class LongLiterals(Relation):
    """The long literals written inside a function body outside the test modules, with the kind of
    each (`sql`, `regex`, `shell`, `token` or `number`) and the name the place it is written gives
    it (`naming`: the keyword or parameter it is the whole value of, or the variable of two words
    or more it is assigned to; a one-word variable such as `value` or `sql` says too little).

    A string is long from 40 characters and counts only when it is code rather than prose: a SQL
    statement, a pattern, a shell line or a single token such as a path or a flag list; a message
    stays where it is said. A number is long from 1000 or from four significant digits (`3600`,
    `1_000_000`, `1.618`); a file mode (`0o755`) or a mask spelled in hex or binary is not."""

    query = r"""
        WITH strings AS (
          SELECT t.path, t.start_line, t.start_column, t.end_line, t.end_column, t.text,
                 t.value AS literal, p.type AS role, p.name,
                 CASE
                   WHEN regexp_matches(t.value, '^\s*(SELECT|WITH|CREATE|INSERT|UPDATE|DELETE|DROP|ALTER|COPY|ATTACH|PRAGMA|INSTALL|LOAD)\b')
                     THEN 'sql'
                   WHEN regexp_matches(t.value, '\\[dswbDSWB]|\(\?|\[\^|\.\*|\.\+') THEN 'regex'
                   WHEN regexp_matches(t.value, '\$\(|\$\{|&&|\|\||>/dev/null|\$[A-Za-z_?]')
                     THEN 'shell'
                   WHEN NOT regexp_matches(t.value, '\s') THEN 'token'
                 END AS kind
          FROM core_text t
          JOIN core_node n ON n.file_id = t.file_id AND n.node_id = t.text_id
          JOIN core_node p ON p.file_id = t.file_id AND p.node_id = t.parent_id
          WHERE t.language = 'python' AND t.kind = 'string' AND t.plain AND t.repeat_id IS NULL
            AND strlen(t.value) >= 40 AND n.function_id IS NOT NULL
            AND t.parent_type IS DISTINCT FROM 'expression_statement'
            AND NOT import_is_test_path(t.path)
        ), numbers AS (
          SELECT f.path, n.start_line, n.start_column, n.end_line, n.end_column, n.name AS text,
                 n.name AS literal, p.type AS role, p.name, 'number' AS kind,
                 try_cast(replace(n.name, '_', '') AS DOUBLE) AS amount,
                 strlen(trim(regexp_replace(
                   CASE WHEN contains(n.name, '.') THEN rtrim(split_part(lower(n.name), 'e', 1), '0')
                        ELSE split_part(lower(n.name), 'e', 1) END,
                   '[^0-9]', '', 'g'), '0')) AS digits
          FROM core_node n
          JOIN core_file f USING (file_id)
          JOIN core_node p ON p.file_id = n.file_id AND p.node_id = n.parent_id
          WHERE f.language = 'python' AND n.type IN ('integer', 'float')
            AND n.function_id IS NOT NULL AND NOT import_is_test_path(f.path)
            AND NOT regexp_matches(lower(n.name), '^0[obx]')
        )
        SELECT path, start_line, start_column, end_line, end_column, text, literal, kind,
               CASE WHEN regexp_matches(name, '^[A-Za-z_]\w*$')
                     AND (role IN ('keyword_argument', 'typed_default_parameter',
                                   'default_parameter')
                          OR (role = 'assignment' AND contains(trim(name, '_'), '_')))
                    THEN name END AS naming
        FROM strings WHERE kind IS NOT NULL
        UNION ALL
        SELECT path, start_line, start_column, end_line, end_column, text, literal, kind,
               CASE WHEN regexp_matches(name, '^[A-Za-z_]\w*$')
                     AND (role IN ('keyword_argument', 'typed_default_parameter',
                                   'default_parameter')
                          OR (role = 'assignment' AND contains(trim(name, '_'), '_')))
                    THEN name END AS naming
        FROM numbers WHERE abs(amount) >= 1000 OR digits >= 4
    """


class LongLiteralInFunction(Rule, id="PY-CONS0004", applicability="review"):
    """A long literal (code-shaped text or a long number) written inside a function, where a
    module constant would name what it is and keep the function to what it does."""

    query = """
        SELECT path, start_line, start_column, end_line, end_column, literal, kind, naming, text
        FROM long_literals()
    """
    message = (
        "`{literal}` is a long {kind} literal inside a function; name it as a module constant"
    )
    # One `name-constants` operation per module: each literal becomes `_NAME`, written as it was
    # after the module's last top-level import, and every occurrence of the same literal reads
    # it. The keyword it is passed as, the parameter it is the default of or the variable it is
    # assigned to names it; a small model names the rest. Left alone: a module with no top-level
    # import, and a literal whose name the module already spells or two meanings would share.
    fix = r"""
        WITH places AS (
          SELECT path, max(end_line) AS after_imports FROM import_binding_nodes
          WHERE role = 'declaration' AND start_column = 0 GROUP BY path
        ), grouped AS (
          SELECT f.path, min(f.finding_key) AS finding_key,
                 list({'path': f.path, 'start_line': f.start_line,
                       'start_column': f.start_column, 'end_line': f.end_line,
                       'end_column': f.end_column, 'text': f.text, 'kind': f.kind,
                       'naming': f.naming}
                      ORDER BY f.start_line, f.start_column) AS literals
          FROM finding f JOIN places p USING (path)
          WHERE f.start_line > p.after_imports
          GROUP BY f.path
        )
        SELECT g.finding_key, 'name-constants' AS operation, g.path, s.content, p.after_imports,
               g.literals
        FROM grouped g JOIN places p USING (path) JOIN source_files s USING (path)
    """
    examples = (
        Example(
            bad=(
                """import subprocess


def gpus():
    return subprocess.run(["nvidia-smi"], timeout=3600, check=True)
"""
            ),
            good=(
                """import subprocess

_TIMEOUT = 3600


def gpus():
    return subprocess.run(["nvidia-smi"], timeout=_TIMEOUT, check=True)
"""
            ),
        ),
        Example(
            bad=(
                """import re


def versions(text):
    version_pattern = r"CUDA (?:UMD )?Version:\\s*(\\d+(?:\\.\\d+)+)"
    return re.findall(version_pattern, text)
"""
            ),
            good=(
                """import re

_VERSION_PATTERN = r"CUDA (?:UMD )?Version:\\s*(\\d+(?:\\.\\d+)+)"


def versions(text):
    version_pattern = _VERSION_PATTERN
    return re.findall(version_pattern, text)
"""
            ),
        ),
    )


class RepeatedPrivateConstant(Rule, id="PY-CONS0005", applicability="review"):
    """One private constant (`_SECONDS_PER_HOUR = 3600`) defined the same way in several modules
    of one top package, where the `constants.py` of the lowest package holding them all would
    define it once. Tests and `constants` modules stay out; values compare as their one-line text.
    The name must say enough for a shared meaning: two words or more, since `_BATCH` or `_JSON` in
    two modules are as often two things, and no version, since two modules at version 1 are two
    versions."""

    query = r"""
        WITH constants AS (
          SELECT s.name, s.path, s.start_line, s.start_column, f.module,
                 regexp_extract(l.text, '^[A-Za-z_]\w*\s*(?::[^=]+)?=\s*(.+?)\s*(?:#.*)?$', 1)
                   AS value
          FROM placed_symbols() s
          JOIN core_file f USING (path)
          JOIN source_lines() l ON l.path = s.path AND l.line = s.start_line
          WHERE s.scope = 'module' AND s.is_constant_assignment
            AND regexp_matches(s.name, '^_[A-Z][A-Z0-9]*_[A-Z0-9_]+$')
            AND NOT contains(s.name, 'VERSION')
            AND NOT import_is_test_path(s.path)
            AND NOT list_contains(string_split(f.module, '.'), 'constants')
        )
        SELECT path, start_line, start_column, name, value, module,
               count(*) OVER (PARTITION BY name, value, split_part(module, '.', 1)) AS modules
        FROM constants WHERE value <> ''
        QUALIFY modules > 1
    """
    message = (
        "`{name} = {value}` is defined the same way in {modules} modules; define it once in the "
        "constants.py of the lowest package holding them"
    )
    # One `share-constants` operation per constant: each module's definition goes, the module
    # reads the public name from the `constants.py` of the lowest package holding every defining
    # module, which defines it once (or already does, the same way). Refused: a value that is not
    # one literal, definitions that differ once parsed, a module already binding the public name
    # or spelling the private one where the syntax tree does not read it, a destination defining
    # it otherwise, and a public name two values of the package would claim (`3600`, `3600.0`).
    fix = r"""
        WITH definers AS (
          SELECT f.name, f.value, split_part(f.module, '.', 1) AS top, f.path, f.module,
                 s.content, min(f.finding_key) AS finding_key,
                 if(ends_with(f.path, '__init__.py'), string_split(f.module, '.'),
                    list_slice(string_split(f.module, '.'), 1, -2)) AS package,
                 string_split(rtrim(regexp_replace(f.path, '[^/]*$', ''), '/'), '/') AS folders
          FROM finding f JOIN source_files s USING (path)
          GROUP BY ALL
        ), anchored AS (
          SELECT d.*, first_value(d.package) OVER w AS anchor,
                 first_value(d.folders) OVER w AS anchor_folders
          FROM definers d
          WINDOW w AS (PARTITION BY d.name, d.value, d.top ORDER BY d.path)
        ), common AS (
          SELECT name, value, top, any_value(anchor) AS anchor,
                 any_value(anchor_folders) AS folders,
                 min(coalesce(list_position(list_transform(
                   range(1, least(len(anchor), len(package)) + 1),
                   lambda k: anchor[k] = package[k]), false) - 1,
                   least(len(anchor), len(package)))) AS depth,
                 min(finding_key) AS finding_key,
                 list({'path': path, 'module': module, 'content': content} ORDER BY path)
                   AS definers
          FROM anchored GROUP BY name, value, top
        ), placed AS (
          SELECT c.*,
                 array_to_string(list_slice(c.anchor, 1, c.depth), '.') AS above,
                 array_to_string(
                   list_slice(c.folders, 1, len(c.folders) - len(c.anchor) + c.depth), '/')
                   AS directory
          FROM common c
        )
        SELECT p.finding_key, 'share-constants' AS operation, p.name, p.definers,
               if(p.directory = '', '', p.directory || '/') || 'constants.py' AS destination,
               if(p.above = '', 'constants', p.above || '.constants') AS destination_module,
               s.content AS destination_content
        FROM placed p
        LEFT JOIN source_files s
          ON s.path = if(p.directory = '', '', p.directory || '/') || 'constants.py'
        WHERE NOT EXISTS (SELECT 1 FROM definers o
                          WHERE ltrim(o.name, '_') = ltrim(p.name, '_') AND o.top = p.top
                            AND o.value <> p.value)
    """
    examples = (
        Example(
            bad="_PER_HOUR = 3600\n\n\ndef late(t):\n    return t > _PER_HOUR\n",
            good="from ..constants import PER_HOUR\n\n\ndef late(t):\n    return t > PER_HOUR\n",
            path="pkg/a/use.py",
            context={
                "pkg/__init__.py": "",
                "pkg/a/__init__.py": "",
                "pkg/b/__init__.py": "",
                "pkg/b/other.py": "_PER_HOUR = 3600\n\n\ndef early(t):\n    return t < _PER_HOUR\n",
            },
        ),
        Example(
            bad=(
                """from typing import Final

_SSH_MODE: Final = 'ssh'


def ssh(mode):
    return mode == _SSH_MODE
"""
            ),
            good=(
                """from typing import Final
from .constants import SSH_MODE


def ssh(mode):
    return mode == SSH_MODE
"""
            ),
            path="pkg/use.py",
            context={
                "pkg/__init__.py": "",
                "pkg/other.py": """from typing import Final

_SSH_MODE: Final = 'ssh'


def local(mode):
    return mode != _SSH_MODE
""",
            },
        ),
    )
