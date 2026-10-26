"""Rules over the doubles of a pytest suite (`integration/README.md`, rules 8 and 9): a refusal
test reads what its double recorded, and no test patches a private name of the project's code."""

from ....catalog import Example, Relation, Rule

_OWN_CODE = {
    "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
    "src/pkg/__init__.py": "",
    "src/pkg/worker.py": "class Worker:\n    spawned = 0\n\n\ndef spawn():\n    pass\n",
}
_RECORDER = """import pytest


class Recorder:
    def __init__(self):
        self.calls = []

    def send(self, message):
        self.calls.append(message)
        raise ValueError(message)


@pytest.fixture
def recorder():
    return Recorder()


"""


class PytestRefusalTests(Relation):
    """The collected tests whose last statement is a `with pytest.raises(...)` block.

    Each row holds the block's place, whether the test `asserts` anything of its own (an assert
    statement or a call named `assert...`, nested functions left out) and the recording `doubles`
    it builds or takes from a fixture: classes of the test side whose methods append to their own
    attributes, and `unittest.mock` objects. A fixture counts when its return annotation names
    such a class or its body builds one.
    """

    evidence_imports = frozenset({"pytest"})
    query = r"""
    WITH tests AS (
      SELECT t.fact_id, t.path, t.test, t.name, t.body, f.file_id,
             split_part(f.module, '.', 1) AS root, n.node_id + n.descendant_count AS last
      FROM test_function_declared t
      JOIN core_file f ON f.path = t.path
      JOIN core_node n ON n.file_id = f.file_id AND n.node_id = t.test
      WHERE t.collected
    ), refusing AS (
      SELECT t.*, w.start_line AS with_line, w.start_column AS with_column
      FROM tests t
      JOIN core_node w ON w.file_id = t.file_id AND w.parent_id = t.body
                      AND w.type = 'with_statement'
      JOIN core_node h ON h.file_id = w.file_id AND h.parent_id = w.node_id
                      AND h.type = 'with_clause'
      WHERE w.sibling_index = (SELECT max(s.sibling_index) FROM core_node s
                               WHERE s.file_id = t.file_id AND s.parent_id = t.body
                                 AND s.type <> 'comment')
        AND EXISTS (SELECT 1 FROM core_call c
                    WHERE c.file_id = t.file_id AND c.qualified_name = 'pytest.raises'
                      AND c.call_id BETWEEN h.node_id AND h.node_id + h.descendant_count)
    ), recorders AS (
      SELECT DISTINCT split_part(f.module, '.', 1) AS root, k.name AS class_name
      FROM core_call c
      JOIN core_file f ON f.file_id = c.file_id
      JOIN core_node k ON k.file_id = c.file_id AND k.node_id = c.class_id
                      AND k.type = 'class_definition'
      WHERE regexp_matches(c.callee, '^self\.\w+\.(append|extend|add|insert|setdefault)$')
    ), fixtures AS (
      SELECT d.file_id, d.def_id, d.name, d.returns, d.size, split_part(f.module, '.', 1) AS root
      FROM core_def d
      JOIN core_decor x ON x.file_id = d.file_id AND x.def_id = d.def_id
                       AND x.name = 'pytest.fixture'
      JOIN core_file f ON f.file_id = d.file_id
    ), doubles AS (
      SELECT r.fact_id, r.test, k.class_name AS double
      FROM refusing r
      JOIN core_call c ON c.file_id = r.file_id AND c.call_id BETWEEN r.test AND r.last
      JOIN recorders k ON k.root = r.root AND k.class_name = regexp_extract(c.callee, '[^.]+$')
      UNION
      SELECT r.fact_id, r.test, regexp_extract(c.callee, '[^.]+$')
      FROM refusing r
      JOIN core_call c ON c.file_id = r.file_id AND c.call_id BETWEEN r.test AND r.last
                      AND c.qualified_name IN ('unittest.mock.Mock', 'unittest.mock.MagicMock',
                                               'unittest.mock.AsyncMock')
      UNION
      SELECT r.fact_id, r.test, k.class_name
      FROM refusing r
      JOIN test_function_params p ON p.fact_id = r.fact_id AND p.fn = r.test
      JOIN fixtures d ON d.name = p.name AND d.root = r.root
      JOIN recorders k ON k.root = d.root
        AND (regexp_matches(coalesce(d.returns, ''), '\b' || k.class_name || '\b')
             OR EXISTS (SELECT 1 FROM core_call c
                        WHERE c.file_id = d.file_id
                          AND c.call_id BETWEEN d.def_id AND d.def_id + d.size
                          AND regexp_extract(c.callee, '[^.]+$') = k.class_name))
    )
    SELECT r.path, r.name, r.with_line, r.with_column,
           EXISTS (SELECT 1 FROM core_node n
                   WHERE n.file_id = r.file_id AND n.node_id BETWEEN r.test AND r.last
                     AND n.function_id = r.test AND n.type = 'assert_statement')
             OR EXISTS (SELECT 1 FROM core_call c
                        WHERE c.file_id = r.file_id AND c.call_id BETWEEN r.test AND r.last
                          AND c.function_id = r.test
                          AND regexp_matches(regexp_extract(c.callee, '[^.]+$'), '^_?assert'))
             AS asserts,
           coalesce((SELECT list(DISTINCT d.double ORDER BY d.double) FROM doubles d
                     WHERE d.fact_id = r.fact_id AND d.test = r.test), []::VARCHAR[]) AS doubles
    FROM refusing r
"""


class RefusalWithoutEffectCheck(Rule, id="PY-TEST0021"):
    """Tests that end on a refusal while holding a recording double, and assert nothing.

    A test that expects a refusal also asserts nothing else happened: the fake's record is empty,
    nothing was written. A test that ends on its `with pytest.raises(...)` block, builds or is given
    a double that records what it is asked, and asserts nothing never looks at that record; a test
    with no such double is left alone, since whether it has an effect to check depends on the code
    under test. There is no fix because which effect to assert, and what the record should hold,
    is knowledge of that code: a generated assertion would pin whatever the code does today.
    """

    query = """
        SELECT path, with_line AS start_line, with_column AS start_column, name AS test,
               array_to_string(doubles, '`, `') AS doubles
        FROM pytest_refusal_tests()
        WHERE NOT asserts AND len(doubles) > 0
    """
    message = "`{test}` ends on a refusal but never reads what `{doubles}` recorded"
    examples = (
        Example(
            bad=_RECORDER
            + """def test_a_refused_message_is_still_recorded(recorder):
    with pytest.raises(ValueError):
        recorder.send("bad")
""",
            good=_RECORDER
            + """def test_a_refused_message_is_still_recorded(recorder):
    with pytest.raises(ValueError):
        recorder.send("bad")
    assert recorder.calls == ["bad"]


def test_a_refusal_with_no_double_has_nothing_to_record():
    with pytest.raises(ValueError):
        int("x")
""",
        ),
        Example(
            bad=_RECORDER
            + """def test_a_double_built_in_the_test_is_a_record_too():
    sender = Recorder()
    with pytest.raises(ValueError):
        sender.send("bad")
""",
            good=_RECORDER
            + """def test_a_double_built_in_the_test_is_a_record_too():
    sender = Recorder()
    with pytest.raises(ValueError):
        sender.send("bad")
    assert sender.calls == ["bad"]
""",
        ),
    )


class PrivatePatchOfOwnCode(Rule, id="PY-TEST0022"):
    """Patches of a private name of the project's own code.

    A double sits at the external seam (process, network, clock, files outside `tmp_path`). A test
    that patches `_name` of a module or class of the project's own distribution
    (`monkeypatch.setattr/setitem/delattr/delitem`, `unittest.mock.patch`, `patch.object`) pins how
    the code is built, not what it does, and a rename breaks it with no behavior changed. A target
    is the project's own when its name is imported from a package that a `pyproject.toml` of the
    project names; an object built in the test, a dunder name and a public name are left alone.
    There is no fix because the public seam to patch instead (a parameter, a collaborator, a
    documented hook) is a design choice of the code under test.
    """

    query = r"""
        WITH own AS (
          SELECT DISTINCT lower(replace(trim(value, '"'''), '-', '_')) AS root
          FROM testing_toml_keys() WHERE table_name = 'project' AND key = 'name'
        ), arguments AS (
          SELECT c.file_id, c.call_id, c.path, c.start_line, c.start_column, c.callee,
                 c.qualified_name, a.type,
                 row_number() OVER (PARTITION BY c.file_id, c.call_id ORDER BY a.ordinal) AS place,
                 count(*) OVER (PARTITION BY c.file_id, c.call_id) AS positionals,
                 syntax_slice(s.content, a.start_line, a.start_column, a.end_line, a.end_column)
                   AS spelled
          FROM core_call c
          JOIN core_call_arg a ON a.file_id = c.file_id AND a.call_id = c.call_id
                              AND NOT a.is_keyword
          JOIN source_files s ON s.path = c.path
          WHERE c.callee IN ('monkeypatch.setattr', 'monkeypatch.delattr', 'monkeypatch.setitem',
                             'monkeypatch.delitem')
             OR c.qualified_name IN ('unittest.mock.patch', 'unittest.mock.patch.object')
        ), shaped AS (
          SELECT file_id, call_id, path, start_line, start_column, callee, qualified_name,
                 positionals, max(spelled) FILTER (place = 1) AS first,
                 max(type) FILTER (place = 1) AS first_type,
                 max(spelled) FILTER (place = 2) AS second,
                 max(type) FILTER (place = 2) AS second_type
          FROM arguments GROUP BY ALL
        ), targets AS (
          SELECT *, CASE WHEN by_path THEN regexp_extract(trim(first, '"'''), '[^.]+$')
                         ELSE trim(second, '"''') END AS attribute,
                 CASE WHEN by_path THEN regexp_replace(trim(first, '"'''), '\.[^.]+$', '')
                      ELSE first END AS target
          FROM (SELECT *, (coalesce(qualified_name, '') = 'unittest.mock.patch'
                           OR (callee = 'monkeypatch.setattr' AND positionals = 2)
                           OR (callee = 'monkeypatch.delattr' AND positionals = 1)) AS by_path
                FROM shaped)
          WHERE CASE WHEN by_path THEN first_type ELSE second_type END = 'string'
        )
        SELECT path, start_line, start_column, callee, attribute, target
        FROM targets t
        WHERE regexp_matches(attribute, '^_[A-Za-z0-9]')
          AND EXISTS (
            SELECT 1 FROM own o
            WHERE o.root = split_part(trim(first, '"'''), '.', 1)
               OR EXISTS (SELECT 1 FROM core_import i
                          WHERE i.file_id = t.file_id
                            AND i.name = regexp_extract(first, '^[A-Za-z_]\w*')
                            AND split_part(i.target, '.', 1) = o.root))
    """
    message = (
        "`{callee}` patches the private `{attribute}` of `{target}`, code of this project: a "
        "rename breaks the test with no behavior changed"
    )
    examples = (
        Example(
            bad="""from pkg.worker import Worker


def test_a_worker_with_a_replaced_spawn_spawns_nothing(monkeypatch):
    monkeypatch.setattr(Worker, "_spawn", lambda self: None)
""",
            good="""from pkg.worker import Worker


class Local:
    _hook = None


def test_a_worker_with_a_replaced_spawn_spawns_nothing(monkeypatch):
    monkeypatch.setattr(Worker, "spawn", lambda self: None)
    monkeypatch.setattr(Local, "_hook", object())
    monkeypatch.setattr(Worker, "__repr__", lambda self: "worker")
    monkeypatch.setattr("shutil.which", lambda name: None)
""",
            context=_OWN_CODE,
        ),
        Example(
            bad="""from unittest.mock import patch


def test_a_dotted_path_names_the_private_function():
    with patch("pkg.worker._spawn"):
        pass
""",
            good="""from unittest.mock import patch

from pkg import worker


def test_a_dotted_path_names_the_public_function():
    with patch("pkg.worker.spawn"), patch.object(worker, "spawn"):
        pass
""",
            context=_OWN_CODE,
        ),
    )
