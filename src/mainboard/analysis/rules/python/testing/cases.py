"""Rules over test cases written out by hand (mcmr's PY-TEST cases group)."""

from ....catalog import Example, Function, Relation, Rule
from ....operations.python.suites import MergeTests


class CommonPrefix(Function):
    """The leading words two names share, words being parted by `_` (`test_f_one` and `test_f_two`
    share `test_f`)."""

    a: str
    b: str

    expression = """array_to_string(
      string_split(a, '_')[1:coalesce(
        list_min(list_filter(
          range(1, least(len(string_split(a, '_')), len(string_split(b, '_'))) + 1),
          lambda i: string_split(a, '_')[i] <> string_split(b, '_')[i])),
        least(len(string_split(a, '_')), len(string_split(b, '_'))) + 1) - 1], '_')"""


class CommonSuffix(Function):
    """The trailing words two names share, words being parted by `_`."""

    a: str
    b: str

    expression = """array_to_string(list_reverse(string_split(common_prefix(
      array_to_string(list_reverse(string_split(a, '_')), '_'),
      array_to_string(list_reverse(string_split(b, '_')), '_')), '_')), '_')"""


class CollectedTestScopes(Relation):
    """The start of the pytest node id of each collected test: its file, then the class holding it
    (`t.py::TestF::`), so `scope || name` is the id."""

    query = """
    SELECT d.path, d.ordinal, d.path || '::' || coalesce((
      SELECT c.name || '::' FROM test_function_nodes h
      JOIN test_function_nodes c ON c.fact_id = h.fact_id AND c.node_id = h.parent_id
        AND c.type = 'class_definition'
      WHERE h.fact_id = d.fact_id AND h.node_id = d.holder AND h.type = 'block'), '') AS scope
    FROM test_function_declared d WHERE d.collected
"""


class MergeableTestGroups(Relation):
    """Each group of two or more collected tests sharing a holder (a module or a class), a body
    shape, marks and what they reach, whose literal cases all differ and none is parametrized or
    property based, with the test they merge into.

    `merged` is the words their names share at both ends (at least `test` and one more, and no
    other group of the holder takes the name), `ids` what sets each name apart in between, and
    `renamed` the id map `old -> merged[id]`, which the fixpoint's collection proof reads.
    """

    query = """
    WITH cases AS (
      SELECT d.path, d.holder, d.test, d.name, s.body_shape, s.literals, r.fixture_names,
             r.reachable_targets,
             coalesce((SELECT list(k.mark ORDER BY k.ordinal) FROM test_function_decorators k
                       WHERE k.fact_id = d.fact_id AND k.fn = d.test), []) AS marks
      FROM test_function_declared d
      JOIN test_function_shapes s ON s.fact_id = d.fact_id AND s.test = d.test
      LEFT JOIN pytest_reaching() r ON r.path = d.path AND r.ordinal = d.ordinal
      WHERE d.collected AND NOT EXISTS (
        SELECT 1 FROM test_function_decorators k WHERE k.fact_id = d.fact_id AND k.fn = d.test
          AND (ends_with(k.mark, 'parametrize') OR ends_with(k.mark, 'given')))
    ), grouped AS (
      SELECT path, holder, body_shape, marks, fixture_names, reachable_targets, count(*) AS size,
             list(test ORDER BY test) AS tests, list(name ORDER BY test) AS names,
             list_reduce(list(name ORDER BY test), lambda a, b: common_prefix(a, b)) AS head
      FROM cases
      GROUP BY path, holder, body_shape, marks, fixture_names, reachable_targets
      HAVING count(*) > 1 AND count(DISTINCT literals) = count(*)
    ), parted AS (
      SELECT *, list_transform(names,
                  lambda n: string_split(n, '_')[len(string_split(head, '_')) + 1:]) AS rests
      FROM grouped
    ), tailed AS (
      SELECT *, greatest(0, least(
               coalesce(len(string_split(nullif(list_reduce(
                 list_transform(rests, lambda r: array_to_string(r, '_')),
                 lambda a, b: common_suffix(a, b)), ''), '_')), 0),
               list_min(list_transform(rests, lambda r: len(r))) - 1)) AS tail
      FROM parted
    ), identified AS (
      SELECT *, head || CASE WHEN tail > 0
                            THEN '_' || array_to_string(rests[1][len(rests[1]) - tail + 1:], '_')
                            ELSE '' END AS merged,
             list_transform(rests, lambda r: array_to_string(r[1:len(r) - tail], '-')) AS ids
      FROM tailed
      WHERE head <> '' AND (len(string_split(head, '_')) > 1 OR tail > 0)
        AND list_bool_and(list_transform(ids, lambda i: i <> ''))
      QUALIFY count(*) OVER (PARTITION BY path, holder, merged) = 1
    )
    SELECT *, array_to_string(list_transform(range(1, size + 1),
                lambda i: names[i] || ' -> ' || merged || '[' || ids[i] || ']'), ', ') AS renamed
    FROM identified
"""


class PytestVectorGroups(Relation):
    """Each case group with how many literal vectors it holds and how many differ."""

    query = """
    SELECT g.fact_id, coalesce(c.vector_count, 0) AS vector_count,
           coalesce(c.distinct_vector_count, 0) AS distinct_vector_count
    FROM test_case_group_groups g LEFT JOIN (
      SELECT record_id, count(*) AS vector_count, count(DISTINCT vector) AS distinct_vector_count
      FROM (
        SELECT any_value(parent_id) FILTER (entry_kind = 'container') AS record_id,
               coalesce(list(string_value ORDER BY ordinal) FILTER (entry_kind = 'value'),
                        []::VARCHAR[]) AS vector
        FROM test_case_group_values WHERE relation = 'groups.literal_vectors'
        GROUP BY container_id HAVING bool_or(entry_kind = 'container'))
      GROUP BY record_id) c USING (record_id)
"""


class ParametrizationCandidateGroupCount(Rule, id="PY-TEST0003", applicability="review"):
    """Groups of tests with a few distinct literal cases that one parametrized test would cover."""

    minimum_cases: int = 3
    maximum_cases: int = 9

    query = f"""
        SELECT f.*, n AS value, m.renamed FROM test_case_group_facts f JOIN (
          SELECT fact_id, count(*) AS n FROM {PytestVectorGroups.name()}()
          WHERE vector_count BETWEEN $minimum_cases AND $maximum_cases
            AND vector_count = distinct_vector_count
          GROUP BY fact_id) USING (fact_id)
        LEFT JOIN (
          SELECT path, string_agg(renamed, ', ' ORDER BY tests[1]) AS renamed
          FROM {MergeableTestGroups.name()}()
          WHERE size BETWEEN $minimum_cases AND $maximum_cases GROUP BY path) m USING (path)
    """
    message = "parametrization candidate group count is {value} for `{fact_id}`"
    change = "parametrized {renamed}"
    fix = f"""
        SELECT f.finding_key, '{MergeTests.name}' AS operation, g.path, g.holder AS node, g.tests,
               g.merged AS name, g.ids
        FROM finding f JOIN {MergeableTestGroups.name()}() g ON g.path = f.path
        WHERE g.size BETWEEN $minimum_cases AND $maximum_cases
    """
    examples = (
        Example(
            bad=(
                """def f(x):
    return x + 1


def test_f_one():
    assert f(1) == 2


def test_f_two():
    assert f(2) == 3


def test_f_three():
    assert f(3) == 4
"""
            ),
            good=(
                """import pytest


def f(x):
    return x + 1


@pytest.mark.parametrize(
    ("case_0", "expected"),
    [
        pytest.param(1, 2, id="one"),
        pytest.param(2, 3, id="two"),
        pytest.param(3, 4, id="three"),
    ],
)
def test_f(case_0, expected):
    assert f(case_0) == expected
"""
            ),
        ),
        Example(
            bad=(
                """def parse(text):
    return text.upper()


def test_parse_a():
    assert parse(text="a") == "A"


def test_parse_b():
    assert parse(text="b") == "B"


def test_parse_c():
    assert parse(text="c") == "C"
"""
            ),
            good=(
                """import pytest


def parse(text):
    return text.upper()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("a", "A", id="a"),
        pytest.param("b", "B", id="b"),
        pytest.param("c", "C", id="c"),
    ],
)
def test_parse(text, expected):
    assert parse(text=text) == expected
"""
            ),
        ),
        Example(
            bad=(
                """import pytest


def f(x):
    return x + 1


def test_f_one():
    assert f(1) == 2


@pytest.mark.slow
def test_f_two():
    assert f(2) == 3


def test_f_three():
    assert f(3) == 4
"""
            ),
            good=(
                """import pytest


def f(x):
    return x + 1


def test_f_one():
    assert f(1) == 2


@pytest.mark.slow
def test_f_two():
    assert f(2) == 3


def test_f_three():
    assert f(3) == 4
"""
            ),
        ),
        Example(
            bad=(
                """import pytest


def f(x):
    return x + 1


@pytest.mark.skipif(f(0) == 1, reason="one")
def test_f_one():
    assert f(1) == 2


@pytest.mark.skipif(f(0) == 2, reason="two")
def test_f_two():
    assert f(2) == 3


@pytest.mark.skipif(f(0) == 3, reason="three")
def test_f_three():
    assert f(3) == 4
"""
            ),
            good=(
                """import pytest


def f(x):
    return x + 1


@pytest.mark.skipif(f(0) == 1, reason="one")
def test_f_one():
    assert f(1) == 2


@pytest.mark.skipif(f(0) == 2, reason="two")
def test_f_two():
    assert f(2) == 3


@pytest.mark.skipif(f(0) == 3, reason="three")
def test_f_three():
    assert f(3) == 4
"""
            ),
        ),
    )


class TestingCasesLoopEdits(Relation):
    """The edits turning a test whose whole body is a loop over literal cases into a parametrized
    test: the decorator above the test, the loop variables as parameters, the loop header gone and
    its body dedented; three edits per loop (`loop`). Only where the rewrite is exact: a one-line
    signature, only `pytest.mark` decorators, a plain class or none, a loop that is the test's last
    and only statement (beside a docstring) over literal cases of the right shape, no `else`,
    `break`, `continue`, `return` or `yield`, no comment in the header, no multi-line string in the
    body, and loop variables the signature does not already name."""

    minimum_cases: int

    query = r"""
    WITH tests AS (
      SELECT t.record_id, t.path, t.name, t.node_span_start_line AS first_line,
             t.node_span_start_column AS indent_column, t.node_span_end_line AS last_line,
             s.fact_id, s.source,
             coalesce((SELECT list(v.string_value ORDER BY v.ordinal) FROM test_function_values v
                       WHERE v.fact_id = t.fact_id AND v.parent_id = t.record_id
                         AND v.relation = 'tests.requested_fixture_names'), []) AS parameters
      FROM test_function_tests t
      JOIN syntax_facts s ON s.path = t.path AND s.kind = 'callable'
        AND s.start_line = t.node_span_start_line AND s.start_column = t.node_span_start_column
      LEFT JOIN syntax_facts k ON k.path = s.path AND k.kind = 'type'
        AND k.qualname = regexp_replace(s.qualname, '\.[^.]+$', '')
      WHERE t.is_collected
        AND NOT EXISTS (SELECT 1 FROM test_function_values v
                        WHERE v.fact_id = t.fact_id AND v.parent_id = t.record_id
                          AND v.relation = 'tests.marks'
                          AND NOT starts_with(v.string_value, 'pytest.mark.'))
        AND (NOT contains(s.qualname, '.') OR regexp_matches(
               regexp_extract(k.source, '(?m)^class[^\n]*'),
               '^class\s+\w+\s*(\(\s*(object)?\s*\))?\s*:'))
    ), defs AS (
      SELECT t.*, l.line AS def_line, regexp_extract(l.text,
               '^(\s*(?:async\s+)?def\s+\w+\s*\()([^()]*)(\)\s*(?:->[^:]*)?:\s*(?:#.*)?)$',
               ['head', 'params', 'tail']) AS sig
      FROM tests t JOIN source_lines() l
        ON l.path = t.path AND l.line BETWEEN t.first_line AND t.last_line
      WHERE regexp_matches(l.text, '^\s*(async\s+)?def\s+' || t.name || '\s*\(')
      QUALIFY row_number() OVER (PARTITION BY t.record_id ORDER BY l.line) = 1
    ), loops AS (
      SELECT d.*, l.ordinal AS loop_ordinal, l.subtree_end AS loop_subtree_end,
             l.start_line AS loop_start, l.start_column AS loop_column, l.end_line AS loop_end
      FROM defs d JOIN syntax_nodes l ON l.fact_id = d.fact_id AND l.kind = 'loop' AND l.depth = 2
        AND l.end_line = d.last_line AND l.start_line > d.def_line
      WHERE d.sig.head <> '' AND NOT contains(d.sig.params, '**')
        AND (SELECT count(*) FROM syntax_nodes b WHERE b.fact_id = d.fact_id AND b.depth = 2
               AND b.start_line > d.def_line) - (SELECT count(*) FROM syntax_nodes b
               JOIN syntax_nodes x ON x.fact_id = b.fact_id AND x.ordinal = b.ordinal + 1
               WHERE b.fact_id = d.fact_id AND b.depth = 2 AND b.start_line > d.def_line
                 AND b.ordinal < l.ordinal AND b.kind = 'effect' AND b.subtree_end = b.ordinal + 2
                 AND x.kind = 'text') = 1
    ), shaped AS (
      SELECT lp.*, tg.kind AS target_kind, tg.name AS target_name, tg.node_id AS target_node,
             it.ordinal AS iter_ordinal, it.subtree_end AS iter_end, it.node_id AS iter_node,
             node_text(lp.source, it.byte_start, it.byte_length) AS iter_text,
             bf.start_line AS body_line, bf.start_column AS body_column, bf.ordinal AS body_ordinal
      FROM loops lp
      JOIN syntax_nodes tg ON tg.fact_id = lp.fact_id AND tg.ordinal = lp.loop_ordinal + 1
      JOIN syntax_nodes it ON it.fact_id = lp.fact_id AND it.ordinal = tg.subtree_end
      JOIN syntax_nodes bf ON bf.fact_id = lp.fact_id AND bf.ordinal = it.subtree_end
      WHERE it.kind = 'collection' AND bf.start_line > it.end_line
        AND bf.start_column > lp.loop_column
    ), named AS (
      SELECT s.*, CASE WHEN s.target_kind = 'name' THEN [s.target_name]
                       ELSE (SELECT list(c.name ORDER BY c.ordinal) FROM syntax_nodes c
                             WHERE c.fact_id = s.fact_id AND c.parent_id = s.target_node
                               AND c.kind = 'name') END AS names,
             (SELECT count(*) FROM syntax_nodes c
              WHERE c.fact_id = s.fact_id AND c.parent_id = s.target_node) AS target_children,
             (SELECT count(*) FROM syntax_nodes c
              WHERE c.fact_id = s.fact_id AND c.parent_id = s.iter_node) AS case_count
      FROM shaped s
    ), checked AS (
      SELECT n.*,
             bool_or(l.line > n.loop_start AND regexp_matches(l.text,
               '\b(break|continue|return|yield|nonlocal|global)\b')) AS risky,
             bool_or(l.line > n.loop_start AND regexp_matches(l.text,
               '^' || repeat(' ', n.loop_column::BIGINT) || 'else\s*:')) AS has_else,
             bool_or(regexp_matches(l.text, '^\s*\t')) AS has_tab,
             bool_or(l.line < n.body_line AND contains(l.text, '#')) AS header_comment,
             bool_or(l.line >= n.body_line AND regexp_matches(l.text,
               '^' || repeat(' ', n.body_column::BIGINT) || 'assert\b')) AS has_assert
      FROM named n JOIN source_lines() l
        ON l.path = n.path AND l.line BETWEEN n.loop_start AND n.loop_end
      GROUP BY ALL
    ), groups AS (
      SELECT c.* FROM checked c
      WHERE (c.target_kind = 'name' OR (c.target_kind = 'collection' AND c.target_children >= 2
                                       AND len(c.names) = c.target_children))
        AND substr(c.iter_text, 1, 1) IN ('[', '(') AND c.case_count >= minimum_cases
        AND len(list_intersect(c.names, c.parameters)) = 0
        AND NOT c.risky AND NOT c.has_else AND NOT c.has_tab AND NOT c.header_comment
        AND c.has_assert AND NOT contains(c.iter_text, '#')
        AND NOT EXISTS (SELECT 1 FROM syntax_nodes x WHERE x.fact_id = c.fact_id
              AND x.ordinal > c.iter_ordinal AND x.ordinal < c.iter_end
              AND x.kind NOT IN ('collection', 'literal', 'text'))
        AND NOT EXISTS (SELECT 1 FROM syntax_nodes x WHERE x.fact_id = c.fact_id
              AND x.ordinal >= c.body_ordinal AND x.ordinal < c.loop_subtree_end
              AND x.kind = 'text' AND x.start_line <> x.end_line)
        AND (c.target_kind = 'name' OR NOT EXISTS (
              SELECT 1 FROM syntax_nodes x
              WHERE x.fact_id = c.fact_id AND x.parent_id = c.iter_node
                AND NOT (x.kind = 'collection' AND (SELECT count(*) FROM syntax_nodes y
                         WHERE y.fact_id = x.fact_id AND y.parent_id = x.node_id)
                         = c.target_children)))
    ), edits AS (
      SELECT g.*, array_to_string(g.names, ', ') AS names_text, rtrim(g.sig.params) AS params_text
      FROM groups g
    )
    SELECT record_id AS loop, path, first_line AS start_line, 0 AS start_column,
           first_line AS end_line, 0 AS end_column,
           repeat(' ', indent_column::BIGINT) || '@pytest.mark.parametrize("' || names_text
             || '", ' || iter_text || ')' || chr(10) AS text,
           'import pytest' AS "import"
    FROM edits
    UNION ALL BY NAME
    SELECT record_id AS loop, path, def_line AS start_line,
           strlen(sig.head) + strlen(params_text) AS start_column, def_line AS end_line,
           strlen(sig.head) + strlen(params_text) AS end_column,
           CASE WHEN params_text = '' THEN names_text
                WHEN right(params_text, 1) = ',' THEN ' ' || names_text
                ELSE ', ' || names_text END AS text
    FROM edits
    UNION ALL BY NAME
    SELECT e.record_id AS loop, e.path, e.loop_start AS start_line, 0 AS start_column,
           e.loop_end + 1 AS end_line, 0 AS end_column,
           (SELECT string_agg(regexp_replace(l.text,
                    '^ {0,' || (e.body_column - e.loop_column) || '}', ''),
                    chr(10) ORDER BY l.line) || chr(10)
            FROM source_lines() l
            WHERE l.path = e.path AND l.line BETWEEN e.body_line AND e.loop_end) AS text
    FROM edits e
"""


class ManualLiteralTestCaseLoopCount(Rule, id="PY-TEST0013", applicability="review"):
    """Loops that walk literal cases and assert inside the loop body."""

    minimum_cases: int = 3

    query = """
        SELECT f.*, n AS value FROM test_case_group_facts f JOIN (
          SELECT fact_id, count(*) AS n FROM test_case_group_loops
          WHERE case_count >= $minimum_cases AND owns_assertion GROUP BY fact_id) USING (fact_id)
    """
    message = "manual literal test case loop count is {value} for `{fact_id}`"
    # The loop becomes `@pytest.mark.parametrize` over the same literal cases: one test per case,
    # reported and selected apart. Only a test that is nothing but the loop is rewritten.
    fix = """
        SELECT f.finding_key, e.* EXCLUDE (loop)
        FROM finding f JOIN testing_cases_loop_edits($minimum_cases) e ON e.path = f.path
    """
    examples = (
        Example(
            bad=(
                """import pytest


def test_upper():
    for raw in ("A", "B", "C"):
        assert raw.isupper()
"""
            ),
            good=(
                """import pytest


@pytest.mark.parametrize("raw", ("A", "B", "C"))
def test_upper(raw):
    assert raw.isupper()
"""
            ),
        ),
    )
