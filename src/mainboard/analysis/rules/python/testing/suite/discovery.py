"""Rules over how pytest finds and isolates tests (mcmr's PY-TEST discovery group)."""

from .....catalog import Example, Rule
from ...constants.testing_suite import PYTEST_TABLE
from .core import pytest_option_fix

_TESTS = {"integration/__init__.py": "", "integration/x/__init__.py": ""}
_CONFTEST = """import pytest

LIMIT = 3


@pytest.fixture
def git():
    return "git"


def committed():
    return 1


class FakeRepo:
    name = "fake"
"""
_CONFTEST_READING = """import pytest


@pytest.fixture
def git():
    return "git"


def committed():
    return git
"""


class ConftestImport(Rule, id="PY-TEST0001", applicability="review"):
    """Imports of `conftest`, which pytest loads on its own."""

    allowed_modules: tuple[str, ...] = ()

    query = """
        SELECT path, start_line, start_column, 'import:' || path || ':' || name AS fact_id,
               imported_name
        FROM core_import
        WHERE (name = 'conftest' OR imported_name = 'conftest'
               OR list_contains(str_split(module, '.'), 'conftest'))
          AND NOT list_contains($allowed_modules::VARCHAR[], module)
    """
    message = "`{fact_id}` imports conftest, which pytest loads on its own"
    # What a module takes by name from a `conftest.py` leaves it for the conftest's folder, a
    # function for `builders.py`, a class for `fakes.py`, anything else for `constants.py`
    # (`move-definitions`, one move for all the names of a destination), and each import of it is
    # respelled. A fixture stays: pytest hands it to a test of the conftest's folder by its
    # name, so an import left holding fixtures alone goes. Left alone, with the reason, what the
    # operation refuses (a helper reading a name of the conftest that stays, an importer binding
    # the module as `import a as b`) and a module of the conftest's own name taken whole
    # (`import conftest`).
    fix = r"""
        WITH named AS (
          SELECT f.finding_key, x.path, x.start_line, x.start_column, x.end_line, x.end_column,
                 x.name, c.path AS conftest, regexp_replace(c.path, '/?[^/]*$', '') AS folder,
                 CASE d.kind WHEN 'function' THEN 'builders' WHEN 'class' THEN 'fakes'
                             ELSE 'constants' END AS home,
                 coalesce(EXISTS (SELECT 1 FROM core_decor k
                                  WHERE k.file_id = d.file_id AND k.def_id = d.def_id
                                    AND regexp_matches(k.name, '(^|\.)fixture$')), false)
                   AS is_fixture
          FROM finding f
          JOIN imported_names() x ON x.path = f.path AND x.start_line = f.start_line
                                 AND x.start_column = f.start_column AND x.name = f.imported_name
          JOIN core_file c ON c.module = x.module AND ends_with(c.path, 'conftest.py')
          LEFT JOIN core_def d ON d.file_id = c.file_id AND d.name = x.name AND d.owner_id IS NULL
        ), moved AS (
          SELECT min(finding_key) AS finding_key, conftest, list(DISTINCT name ORDER BY name) AS names,
                 module_path(folder, home) AS destination
          FROM named WHERE NOT is_fixture GROUP BY conftest, folder, home
        ), dropped AS (
          SELECT min(finding_key) AS finding_key, path, start_line, start_column, end_line, end_column
          FROM named GROUP BY path, start_line, start_column, end_line, end_column
          HAVING bool_and(is_fixture AND starts_with(path, coalesce(nullif(folder, '') || '/', '')))
        )
        SELECT m.finding_key, 'move-definitions' AS operation, r.*
        FROM moved m, move_row(m.conftest, m.names, m.destination) r
        UNION ALL BY NAME
        SELECT finding_key, path, start_line, start_column, end_line, end_column, '' AS text
        FROM dropped
    """
    examples = (
        Example(
            bad="from ..conftest import committed, git\n\n\ndef test_it(git):\n    assert committed()\n",
            good="from ..builders import committed\n\n\ndef test_it(git):\n    assert committed()\n",
            path="integration/x/test_a.py",
            context={**_TESTS, "integration/conftest.py": _CONFTEST},
        ),
        Example(
            bad="from ..conftest import FakeRepo\n\n\ndef test_it():\n    assert FakeRepo.name\n",
            good="from ..fakes import FakeRepo\n\n\ndef test_it():\n    assert FakeRepo.name\n",
            path="integration/x/test_a.py",
            context={**_TESTS, "integration/conftest.py": _CONFTEST},
        ),
        Example(
            bad="from ..conftest import LIMIT\n\n\ndef test_it():\n    assert LIMIT\n",
            good="from ..constants import LIMIT\n\n\ndef test_it():\n    assert LIMIT\n",
            path="integration/x/test_a.py",
            context={**_TESTS, "integration/conftest.py": _CONFTEST},
        ),
        Example(
            bad="from ..conftest import committed, git\n\n\ndef test_it(git):\n    assert committed()\n",
            good="from ..conftest import committed, git\n\n\ndef test_it(git):\n    assert committed()\n",
            path="integration/x/test_a.py",
            context={**_TESTS, "integration/conftest.py": _CONFTEST_READING},
        ),
    )


class LegacyTmpdirFixtureCount(Rule, id="PY-TEST0002", applicability="review"):
    """Collected tests that request the legacy `tmpdir` fixtures instead of `tmp_path`."""

    query = """
        SELECT f.*, n AS value FROM test_function_facts f JOIN (
          SELECT fact_id, count(*) AS n FROM (
            SELECT DISTINCT fact_id, parent_id, string_value FROM test_function_values
            WHERE relation IN ('tests.fixture_names', 'tests.requested_fixture_names')
              AND string_value IN ('tmpdir', 'tmpdir_factory')
              AND (fact_id, parent_id) IN
                  (SELECT fact_id, record_id FROM test_function_tests WHERE is_collected))
          GROUP BY fact_id) USING (fact_id)
    """
    message = "legacy tmpdir fixture count is {value} for `{fact_id}`"
    # Every test of the file taking `tmpdir` moves to `tmp_path` when each use is one the pathlib
    # API spells the same (`str(tmpdir)`, `.strpath`, `.join(x)` with one atom argument and its
    # `.write`, `.read` and `.strpath`, a variable assigned once from a join), the file whole or
    # not at all. One `rewrite-node` row per node: a rename, a call replaced by `tmp_path / x`
    # with its callee captured to drop it, and a method renamed.
    fix = r"""
        WITH funcs AS (
          SELECT f.finding_key, f.path, p.file_id, d.node_id AS fn, b.node_id AS body,
                 b.descendant_count AS size, prm.node_id AS param
          FROM finding f
          JOIN core_file p ON p.path = f.path
          JOIN core_node d ON d.file_id = p.file_id AND d.type = 'function_definition'
          JOIN core_node ps ON ps.file_id = p.file_id AND ps.parent_id = d.node_id
            AND ps.type = 'parameters'
          JOIN core_node prm ON prm.file_id = p.file_id AND prm.parent_id = ps.node_id
            AND prm.type = 'identifier' AND prm.name = 'tmpdir'
          JOIN core_node b ON b.file_id = p.file_id AND b.parent_id = d.node_id
            AND b.type = 'block'
        ), aliases AS (
          SELECT DISTINCT u.file_id, u.fn, t.name AS alias
          FROM funcs u
          JOIN core_node a ON a.file_id = u.file_id AND a.type = 'assignment'
            AND a.children_count = 3 AND a.node_id BETWEEN u.body AND u.body + u.size
          JOIN core_node t ON t.file_id = u.file_id AND t.parent_id = a.node_id
            AND t.sibling_index = 0 AND t.type = 'identifier'
          JOIN core_node j ON j.file_id = u.file_id AND j.parent_id = a.node_id
            AND j.sibling_index = 2 AND j.type = 'call'
          JOIN core_node r ON r.file_id = u.file_id AND r.parent_id = j.node_id
            AND r.sibling_index = 0 AND r.name = 'tmpdir.join'
        ), around AS (
          SELECT u.finding_key, u.path, u.file_id, u.fn, n.node_id AS node, n.name,
                 n.sibling_index AS at, p1.node_id AS p1, p1.type AS t1, p1.name AS m1,
                 p1.children_count AS c1, p1.sibling_index AS at1, p2.node_id AS p2,
                 p2.type AS t2, p2.name AS m2, p2.sibling_index AS at2, p3.node_id AS p3,
                 p3.type AS t3, p3.name AS m3, p3.children_count AS c3, p4.node_id AS p4,
                 p4.type AS t4, p4.name AS m4, k2.type AS callee2, k2.name AS callee2_name,
                 k4.type AS callee4, k4.name AS callee4_name, a2.children_count AS args2,
                 a4.children_count AS args4, x.node_id AS argument, x.type AS argument_type,
                 e.node_id AS method, r.name AS assigned_from
          FROM funcs u
          JOIN core_node n ON n.file_id = u.file_id AND n.type = 'identifier'
            AND n.node_id BETWEEN u.body AND u.body + u.size
            AND (n.name = 'tmpdir' OR EXISTS (SELECT 1 FROM aliases s
                                              WHERE s.file_id = u.file_id AND s.fn = u.fn AND s.alias = n.name))
          JOIN core_node p1 ON p1.file_id = u.file_id AND p1.node_id = n.parent_id
          LEFT JOIN core_node p2 ON p2.file_id = u.file_id AND p2.node_id = p1.parent_id
          LEFT JOIN core_node p3 ON p3.file_id = u.file_id AND p3.node_id = p2.parent_id
          LEFT JOIN core_node p4 ON p4.file_id = u.file_id AND p4.node_id = p3.parent_id
          LEFT JOIN core_node k2 ON k2.file_id = u.file_id AND k2.parent_id = p2.node_id
            AND k2.sibling_index = 0
          LEFT JOIN core_node k4 ON k4.file_id = u.file_id AND k4.parent_id = p4.node_id
            AND k4.sibling_index = 0
          LEFT JOIN core_node a2 ON a2.file_id = u.file_id AND a2.parent_id = p2.node_id
            AND a2.type = 'argument_list'
          LEFT JOIN core_node a4 ON a4.file_id = u.file_id AND a4.parent_id = p4.node_id
            AND a4.type = 'argument_list'
          LEFT JOIN core_node x ON x.file_id = u.file_id AND x.parent_id = a2.node_id
            AND x.sibling_index = 1 AND a2.children_count = 3
          LEFT JOIN core_node e ON e.file_id = u.file_id AND e.parent_id = p1.node_id
            AND e.sibling_index = 1
          LEFT JOIN core_node s ON s.file_id = u.file_id AND s.parent_id = p1.node_id
            AND s.sibling_index = 2 AND p1.type = 'assignment'
          LEFT JOIN core_node r ON r.file_id = u.file_id AND r.parent_id = s.node_id
            AND r.sibling_index = 0 AND s.type = 'call'
        ), shaped AS (
          SELECT o.*, CASE
            WHEN o.name = 'tmpdir' THEN CASE
              WHEN o.t1 = 'argument_list' AND o.c1 = 3 AND o.t2 = 'call'
                   AND o.callee2 = 'identifier' AND o.callee2_name = 'str' THEN 'rename'
              WHEN o.t1 = 'attribute' AND o.m1 = 'tmpdir.strpath' AND o.at = 0 THEN 'strpath'
              WHEN o.t1 = 'attribute' AND o.m1 = 'tmpdir.join' AND o.at = 0 AND o.t2 = 'call'
                   AND o.at1 = 0 AND o.argument_type IN ('string', 'identifier', 'attribute',
                                                         'call', 'subscript') THEN CASE
                WHEN o.t3 = 'attribute' AND o.at2 = 0 AND ends_with(o.m3, '.write')
                     AND o.t4 = 'call' AND o.args4 = 3 THEN 'join_write'
                WHEN o.t3 = 'attribute' AND o.at2 = 0 AND ends_with(o.m3, '.read')
                     AND o.t4 = 'call' AND o.args4 = 2 THEN 'join_read'
                WHEN o.t3 = 'attribute' AND o.at2 = 0 AND ends_with(o.m3, '.strpath')
                  THEN 'join_strpath'
                WHEN o.t3 = 'argument_list' AND o.c3 = 3 AND o.t4 = 'call'
                     AND o.callee4 = 'identifier' AND o.callee4_name = 'str' THEN 'join_text'
                WHEN o.t3 = 'assignment' AND o.at2 = 2 AND o.t4 = 'expression_statement'
                  THEN 'join_kept'
                ELSE 'other' END
              ELSE 'other' END
            WHEN o.t1 = 'assignment' AND o.at = 0 AND o.assigned_from = 'tmpdir.join'
              THEN 'declared'
            WHEN o.t1 = 'attribute' AND o.at = 0 AND o.t2 = 'call' AND o.at1 = 0 THEN CASE
              WHEN o.m1 = o.name || '.write' AND o.args2 = 3 THEN 'alias_write'
              WHEN o.m1 = o.name || '.read' AND o.args2 = 2 THEN 'alias_read'
              WHEN o.m1 = o.name || '.exists' AND o.args2 = 2 THEN 'alias_kept'
              ELSE 'other' END
            WHEN o.t1 = 'attribute' AND o.at = 0 AND o.m1 = o.name || '.strpath'
              THEN 'alias_strpath'
            WHEN o.t1 = 'argument_list' AND o.c1 = 3 AND o.t2 = 'call'
                 AND o.callee2 = 'identifier' AND o.callee2_name = 'str' THEN 'alias_kept'
            ELSE 'other' END AS kind
          FROM around o
        ), judged AS (
          SELECT u.finding_key, u.fn FROM funcs u
          WHERE NOT EXISTS (SELECT 1 FROM shaped s WHERE s.file_id = u.file_id AND s.fn = u.fn
                            AND s.kind = 'other')
            AND NOT EXISTS (SELECT 1 FROM aliases a WHERE a.file_id = u.file_id AND a.fn = u.fn AND (
                  SELECT count(*) FROM shaped s
                  WHERE s.file_id = u.file_id AND s.fn = u.fn AND s.name = a.alias
                    AND s.kind = 'declared') <> 1)
            AND NOT EXISTS (SELECT 1 FROM core_node n
                            WHERE n.file_id = u.file_id AND n.type = 'identifier'
                              AND n.name = 'tmp_path' AND n.node_id BETWEEN u.fn AND u.body + u.size)
        ), whole AS (
          SELECT u.finding_key FROM funcs u LEFT JOIN judged j USING (finding_key, fn)
          GROUP BY u.finding_key, u.file_id HAVING bool_and(j.fn IS NOT NULL)
             AND NOT EXISTS (SELECT 1 FROM core_node n
                             WHERE n.file_id = u.file_id AND n.name = 'tmpdir_factory')
        ), written AS (
          SELECT u.finding_key, u.path, u.param AS node, 'tmp_path' AS template
          FROM funcs u JOIN whole USING (finding_key)
          UNION ALL
          SELECT s.finding_key, s.path, if(s.kind = 'rename', s.node, s.p1),
                 CASE s.kind WHEN 'rename' THEN 'tmp_path' WHEN 'strpath' THEN 'str(tmp_path)'
                             ELSE 'str(' || s.name || ')' END
          FROM shaped s JOIN whole USING (finding_key)
          WHERE s.kind IN ('rename', 'strpath', 'alias_strpath')
          UNION ALL
          SELECT s.finding_key, s.path,
                 if(s.kind LIKE 'join\_%' ESCAPE '\', m.node_id, s.method),
                 if(s.kind IN ('join_write', 'alias_write'), 'write_text', 'read_text')
          FROM shaped s JOIN whole USING (finding_key)
          LEFT JOIN core_node m ON m.file_id = s.file_id AND m.parent_id = s.p3
            AND m.sibling_index = 1
          WHERE s.kind IN ('join_write', 'join_read', 'alias_write', 'alias_read')
        )
        SELECT finding_key, 'rewrite-node' AS operation, path, node, MAP {'OLD': [node]} AS captures,
               template
        FROM written
        UNION ALL BY NAME
        SELECT s.finding_key, 'rewrite-node' AS operation, s.path,
               if(s.kind = 'join_strpath', s.p3, s.p2) AS node,
               CASE WHEN s.kind = 'join_strpath'
                    THEN MAP {'CALLEE': [s.p1], 'ARGUMENT': [s.argument], 'TAIL': [m.node_id]}
                    ELSE MAP {'CALLEE': [s.p1], 'ARGUMENT': [s.argument]} END AS captures,
               CASE s.kind WHEN 'join_strpath' THEN 'str(tmp_path / $ARGUMENT)'
                           WHEN 'join_write' THEN '(tmp_path / $ARGUMENT)'
                           WHEN 'join_read' THEN '(tmp_path / $ARGUMENT)'
                           ELSE 'tmp_path / $ARGUMENT' END AS template
        FROM shaped s JOIN whole USING (finding_key)
        LEFT JOIN core_node m ON m.file_id = s.file_id AND m.parent_id = s.p3
          AND m.sibling_index = 1
        WHERE s.kind LIKE 'join\_%' ESCAPE '\'
    """
    examples = (
        Example(
            bad=(
                """def test_write(tmpdir):
    p = tmpdir.join("x")
    p.write("data")
    assert p.read() == "data"
"""
            ),
            good=(
                """def test_write(tmp_path):
    p = tmp_path / "x"
    p.write_text("data")
    assert p.read_text() == "data"
"""
            ),
        ),
        Example(
            bad=(
                """def test_paths(tmpdir):
    tmpdir.join("y").write("z")
    assert str(tmpdir.join("v")) and tmpdir.strpath
    assert tmpdir.join("w").strpath
"""
            ),
            good=(
                """def test_paths(tmp_path):
    (tmp_path / "y").write_text("z")
    assert str(tmp_path / "v") and str(tmp_path)
    assert str(tmp_path / "w")
"""
            ),
        ),
        Example(
            bad=(
                """import pytest


@pytest.fixture
def home(tmpdir):
    return str(tmpdir)


def test_home(home):
    assert home
"""
            ),
            good=(
                """import pytest


@pytest.fixture
def home(tmp_path):
    return str(tmp_path)


def test_home(home):
    assert home
"""
            ),
        ),
        Example(
            bad=(
                """def test_made(tmpdir):
    tmpdir.ensure("x")


def test_joined(tmpdir):
    assert tmpdir.join("y").check()
"""
            ),
            good=(
                """def test_made(tmpdir):
    tmpdir.ensure("x")


def test_joined(tmpdir):
    assert tmpdir.join("y").check()
"""
            ),
        ),
        Example(
            bad=(
                """def test_passed(tmpdir):
    p = tmpdir.join("x")
    helper(tmpdir)
    assert p.read()
"""
            ),
            good=(
                """def test_passed(tmpdir):
    p = tmpdir.join("x")
    helper(tmpdir)
    assert p.read()
"""
            ),
        ),
        Example(
            bad=(
                """def test_alias(tmpdir):
    p = tmpdir.join("x")
    p.write("data")
    p.remove()
"""
            ),
            good=(
                """def test_alias(tmpdir):
    p = tmpdir.join("x")
    p.write("data")
    p.remove()
"""
            ),
        ),
        Example(
            bad=(
                """def test_dir(tmpdir):
    assert str(tmpdir)


def test_factory(tmpdir_factory):
    base = tmpdir_factory.mktemp("d")
    assert str(base)
"""
            ),
            good=(
                """def test_dir(tmpdir):
    assert str(tmpdir)


def test_factory(tmpdir_factory):
    base = tmpdir_factory.mktemp("d")
    assert str(base)
"""
            ),
        ),
        Example(
            bad=(
                """def test_named(tmpdir, tmp_path):
    assert str(tmpdir) and tmp_path
"""
            ),
            good=(
                """def test_named(tmpdir, tmp_path):
    assert str(tmpdir) and tmp_path
"""
            ),
        ),
    )


class PytestImportIsolation(Rule, id="PY-TEST0005", applicability="review"):
    """Pytest import modes that mutate `sys.path` during collection instead of `importlib`."""

    query = """
        SELECT *, '`' || path || '` uses pytest import mode `' || import_mode || '`, which '
          || CASE WHEN import_mode IN ('append', 'prepend')
                  THEN 'changes `sys.path` during collection'
                  ELSE 'pytest does not recognize' END AS message
        FROM test_suite_facts WHERE import_mode <> 'importlib'
    """
    # `importlib` leaves `sys.path` alone, so tests that import each other or helper modules
    # through the rootdir stop finding them: for review. An `--import-mode` flag in `addopts`
    # overrides the key, so a table that sets one is left.
    fix = pytest_option_fix(
        "import_mode",
        value='"importlib"',
        where=f"""NOT EXISTS (SELECT 1 FROM testing_toml_keys() k WHERE k.path = f.path
          AND k.table_name = '{PYTEST_TABLE}' AND contains(k.text, 'import-mode'))""",
    )
    examples = (
        Example(
            bad='[tool.pytest.ini_options]\nimport_mode = "append"\n',
            good='[tool.pytest.ini_options]\nimport_mode = "importlib"\n',
        ),
    )
