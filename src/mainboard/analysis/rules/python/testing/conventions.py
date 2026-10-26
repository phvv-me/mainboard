"""Rules over how a pytest suite declares its properties and cases (`integration/README.md`, rules
5 and 7): one Hypothesis profile that tests repeat nothing of, and kebab-case parametrize ids."""

from ....catalog import Example, Relation, Rule

_PROFILE = {
    "conftest.py": """from hypothesis import settings

settings.register_profile("mb", database=None, deadline=None)
settings.load_profile("mb")
"""
}


class HypothesisProfileKeywords(Relation):
    """The keywords of each Hypothesis profile that a `conftest.py` registers and also loads, as
    the source spells their values, with the directory (`home`) whose tests that `conftest.py`
    governs. A profile registered with no keyword has one row with a null `keyword`."""

    evidence_imports = frozenset({"hypothesis"})
    query = r"""
    WITH named AS (
      SELECT c.file_id, c.call_id, c.path, c.qualified_name, trim(a.name, '"''') AS profile
      FROM core_call c
      JOIN core_call_arg a ON a.file_id = c.file_id AND a.call_id = c.call_id
                          AND a.ordinal = 0 AND a.type = 'string'
      WHERE c.qualified_name IN ('hypothesis.settings.register_profile',
                                 'hypothesis.settings.load_profile')
        AND regexp_matches(c.path, '(^|/)conftest\.py$')
    )
    SELECT regexp_replace(r.path, 'conftest\.py$', '') AS home, r.profile, k.name AS keyword,
           syntax_slice(s.content, v.start_line, v.start_column, v.end_line, v.end_column) AS value
    FROM named r
    JOIN named l ON l.file_id = r.file_id AND l.profile = r.profile
                AND l.qualified_name = 'hypothesis.settings.load_profile'
    JOIN source_files s ON s.path = r.path
    LEFT JOIN core_call_arg k ON k.file_id = r.file_id AND k.call_id = r.call_id
                             AND k.is_keyword AND k.name <> 'parent'
    LEFT JOIN core_node v ON v.file_id = k.file_id AND v.parent_id = k.node_id
                         AND v.sibling_index = 2
    WHERE r.qualified_name = 'hypothesis.settings.register_profile'
"""


class HypothesisSettingsArguments(Relation):
    """The arguments of each `@settings(...)` decorator that a loaded profile governs (the
    nearest `conftest.py` above the file that registers and loads one), each with its place, the
    decorator's lines, the source spelling of its `value` and whether the profile already sets
    that keyword to that value (`redundant`)."""

    stored = True
    evidence_imports = frozenset({"hypothesis"})
    query = r"""
    WITH calls AS (
      SELECT c.file_id, c.call_id, c.path, c.callee, c.start_line, c.start_column,
             d.start_line AS decorator_line, d.end_line AS decorator_end
      FROM core_call c
      JOIN core_node d ON d.file_id = c.file_id AND d.node_id = c.parent_id AND d.type = 'decorator'
      WHERE c.qualified_name = 'hypothesis.settings'
    ), profiles AS (
      SELECT DISTINCT home, profile FROM hypothesis_profile_keywords()
    ), governed AS (
      SELECT c.*, p.home, p.profile
      FROM calls c JOIN profiles p ON starts_with(c.path, p.home)
      QUALIFY row_number() OVER (PARTITION BY c.file_id, c.call_id
                                 ORDER BY length(p.home) DESC, p.profile) = 1
    ), spelled AS (
      SELECT g.*, a.ordinal, CASE WHEN a.is_keyword THEN a.name END AS keyword,
             a.start_line AS arg_start_line, a.start_column AS arg_start_column,
             a.end_line AS arg_end_line, a.end_column AS arg_end_column,
             syntax_slice(s.content, v.start_line, v.start_column, v.end_line, v.end_column)
               AS value
      FROM governed g
      JOIN core_call_arg a ON a.file_id = g.file_id AND a.call_id = g.call_id
      JOIN source_files s ON s.path = g.path
      LEFT JOIN core_node v ON v.file_id = a.file_id AND v.parent_id = a.node_id
                           AND v.sibling_index = 2 AND a.is_keyword
    )
    SELECT s.* EXCLUDE (file_id),
           EXISTS (SELECT 1 FROM hypothesis_profile_keywords() p
                   WHERE p.home = s.home AND p.profile = s.profile AND p.keyword = s.keyword
                     AND p.value = s.value) AS redundant
    FROM spelled s
"""


class HypothesisSettingsRepeatProfile(Rule, id="PY-TEST0020", applicability="safe"):
    """`@settings(...)` keywords that repeat what the loaded Hypothesis profile already sets.

    The project's `conftest.py` registers and loads one profile (`database=None, deadline=None`),
    so a test states only what it changes (`max_examples`). A keyword given the profile's own value
    does nothing. The rule is silent where no `conftest.py` registers and loads a profile.
    """

    query = """
        SELECT path, call_id, start_line, start_column, any_value(profile) AS profile,
               string_agg('`' || keyword || '=' || value || '`', ', ' ORDER BY ordinal) AS repeated
        FROM hypothesis_settings_arguments() WHERE redundant
        GROUP BY path, call_id, start_line, start_column
    """
    message = "`@settings` repeats {repeated} that the loaded profile `{profile}` already sets"
    # A run of redundant arguments goes as one piece: its lines when it fills them, else up to the
    # next argument, else back from the one before. A decorator left with no argument goes whole,
    # and with it the `settings` import when nothing else uses it; the decorators and that import
    # of one file are one group, so a pass never strands a name.
    fix = r"""
        WITH args AS (
          SELECT *, ordinal - row_number() OVER (PARTITION BY path, call_id, redundant
                                                 ORDER BY ordinal) AS island,
                 bool_and(redundant) OVER (PARTITION BY path, call_id) AS whole
          FROM hypothesis_settings_arguments()
        ), runs AS (
          SELECT path, call_id, callee, decorator_line, decorator_end, whole,
                 min(ordinal) AS first, max(ordinal) AS last,
                 arg_min(arg_start_line, ordinal) AS start_line,
                 arg_min(arg_start_column, ordinal) AS start_column,
                 arg_max(arg_end_line, ordinal) AS end_line,
                 arg_max(arg_end_column, ordinal) AS end_column
          FROM args WHERE redundant
          GROUP BY path, call_id, callee, decorator_line, decorator_end, whole, island
        ), around AS (
          SELECT r.*, p.arg_end_line AS before_line, p.arg_end_column AS before_column,
                 n.arg_start_line AS after_line, n.arg_start_column AS after_column,
                 trimmed(imports_before(b.text, r.start_column)) = ''
                   AND regexp_matches(imports_from(e.text, r.end_column), '^\s*,?\s*(#.*)?$')
                   AS fills_lines
          FROM runs r
          LEFT JOIN args p ON p.path = r.path AND p.call_id = r.call_id AND p.ordinal = r.first - 1
          LEFT JOIN args n ON n.path = r.path AND n.call_id = r.call_id AND n.ordinal = r.last + 1
          JOIN source_lines() b ON b.path = r.path AND b.line = r.start_line
          JOIN source_lines() e ON e.path = r.path AND e.line = r.end_line
        ), deletions AS (
          SELECT path, call_id,
                 CASE WHEN whole THEN decorator_line WHEN fills_lines THEN start_line
                      WHEN after_line IS NOT NULL THEN start_line ELSE before_line END
                   AS start_line,
                 CASE WHEN whole OR fills_lines THEN 0
                      WHEN after_line IS NOT NULL THEN start_column ELSE before_column END
                   AS start_column,
                 CASE WHEN whole THEN decorator_end + 1 WHEN fills_lines THEN end_line + 1
                      WHEN after_line IS NOT NULL THEN after_line ELSE end_line END AS end_line,
                 CASE WHEN whole OR fills_lines THEN 0
                      WHEN after_line IS NOT NULL THEN after_column ELSE end_column END
                   AS end_column
          FROM around
        ), whole_decorators AS (
          SELECT path, callee, count(*) AS dropped FROM runs WHERE whole GROUP BY path, callee
        ), unused AS (
          SELECT w.path, i.binding_id AS member_id
          FROM whole_decorators w
          JOIN import_binding_facts i ON i.path = w.path AND i.module = 'hypothesis'
                                     AND i.imported_name = 'settings' AND i.name = w.callee
          WHERE i.reference_count = w.dropped AND NOT i.is_reexported
            AND NOT i.has_documented_side_effect
        ), gone AS (
          SELECT u.path, r.start_line, r.start_column, r.end_line, r.end_column
          FROM unused u
          JOIN imports_member_removals((SELECT list(member_id) FROM unused)) r USING (member_id)
          WHERE r.feasible
        ), keyed AS (
          SELECT path, min(finding_key) AS first_key FROM finding GROUP BY path
        )
        SELECT CASE WHEN g.path IS NOT NULL THEN k.first_key ELSE f.finding_key END AS finding_key,
               d.path, d.start_line, d.start_column, d.end_line, d.end_column, '' AS text
        FROM deletions d
        JOIN finding f ON f.path = d.path AND f.call_id = d.call_id
        JOIN keyed k ON k.path = d.path
        LEFT JOIN (SELECT DISTINCT path FROM gone WHERE start_line IS NOT NULL) g
          ON g.path = d.path
        UNION ALL
        SELECT k.first_key, g.path, g.start_line, g.start_column, g.end_line, g.end_column, ''
        FROM gone g JOIN keyed k ON k.path = g.path WHERE g.start_line IS NOT NULL
    """
    examples = (
        Example(
            bad="""from hypothesis import given, settings
from hypothesis import strategies as st


@settings(database=None, max_examples=50, deadline=None)
@given(st.integers())
def test_a(number):
    assert number == number


@settings(
    database=None,
    deadline=500,
)
@given(st.integers())
def test_b(number):
    assert number == number
""",
            good="""from hypothesis import given, settings
from hypothesis import strategies as st


@settings(max_examples=50)
@given(st.integers())
def test_a(number):
    assert number == number


@settings(
    deadline=500,
)
@given(st.integers())
def test_b(number):
    assert number == number
""",
            context=_PROFILE,
        ),
        Example(
            bad="""from hypothesis import given, settings
from hypothesis import strategies as st


@settings(deadline=None)
@given(st.integers())
def test_a(number):
    assert number == number


@settings(
    database=None,
    deadline=None,
)
@given(st.text())
def test_b(text):
    assert text == text
""",
            good="""from hypothesis import given
from hypothesis import strategies as st


@given(st.integers())
def test_a(number):
    assert number == number


@given(st.text())
def test_b(text):
    assert text == text
""",
            context=_PROFILE,
        ),
    )


class ParametrizeIdStyle(Rule, id="PY-TEST0023", applicability="review"):
    """Parametrize ids that are not lowercase kebab-case.

    Every `id="..."` of `pytest.param(...)` and every string of an `ids=[...]` list describes the
    situation (`a-symlink-leaving-the-bundle`) and matches `^[a-z0-9]+(-[a-z0-9]+)*$`. Only plain
    string literals are read: an f-string, a prefixed or triple-quoted string, or an id a function
    computes is left alone.
    """

    query = r"""
        WITH written AS (
          SELECT c.path, c.parent_id AS holder, v.start_line, v.start_column, v.end_line,
                 v.end_column
          FROM core_call c
          JOIN core_call_arg k ON k.file_id = c.file_id AND k.call_id = c.call_id
                              AND k.is_keyword AND k.name = 'id'
          JOIN core_node v ON v.file_id = k.file_id AND v.parent_id = k.node_id
                          AND v.sibling_index = 2 AND v.type = 'string'
          WHERE c.qualified_name = 'pytest.param'
          UNION ALL
          SELECT c.path, l.node_id, e.start_line, e.start_column, e.end_line, e.end_column
          FROM core_call c
          JOIN core_call_arg k ON k.file_id = c.file_id AND k.call_id = c.call_id
                              AND k.is_keyword AND k.name = 'ids'
          JOIN core_node l ON l.file_id = k.file_id AND l.parent_id = k.node_id
                          AND l.sibling_index = 2 AND l.type IN ('list', 'tuple')
          JOIN core_node e ON e.file_id = l.file_id AND e.parent_id = l.node_id
                          AND e.type = 'string'
          WHERE c.qualified_name IN ('pytest.mark.parametrize', 'pytest.fixture')
        ), plain AS (
          SELECT *, substr(literal, 1, 1) AS quote, substr(literal, 2, length(literal) - 2) AS id
          FROM (SELECT w.*, syntax_slice(s.content, w.start_line, w.start_column, w.end_line,
                                         w.end_column) AS literal
                FROM written w JOIN source_files s USING (path))
          WHERE regexp_matches(literal, '^"[^"\\\n]*"$')
             OR regexp_matches(literal, '^''[^''\\\n]*''$')
        ), kebab AS (
          SELECT *, trim(regexp_replace(regexp_replace(regexp_replace(lower(id), '[\s_./]+', '-',
                   'g'), '[^a-z0-9-]', '', 'g'), '-+', '-', 'g'), '-') AS normal
          FROM plain
        ), counted AS (
          SELECT *, count(*) OVER (PARTITION BY path, holder, normal) AS same FROM kebab
        )
        SELECT path, start_line, start_column, end_line, end_column, id, quote, normal,
               normal <> '' AND same = 1 AND NOT regexp_matches(id, '[^\x00-\x7F]') AS fixable
        FROM counted WHERE NOT regexp_matches(id, '^[a-z0-9]+(-[a-z0-9]+)*$')
    """
    message = "parametrize id `{id}` is not lowercase kebab-case"
    # The id becomes its kebab-case spelling: lowercase, runs of whitespace, underscores, dots and
    # slashes one hyphen, other punctuation dropped, hyphens trimmed. An id that spells nothing
    # after that, one with a letter outside ASCII (dropping it would change the word), or one
    # whose spelling another id of the same table already has (pytest refuses duplicates) keeps
    # its name for a person to write. Review because the test ids change, which the gate's
    # baseline lists.
    fix = """
        SELECT finding_key, path, start_line, start_column, end_line, end_column,
               quote || normal || quote AS text
        FROM finding WHERE fixable
    """
    examples = (
        Example(
            bad="""import pytest


@pytest.mark.parametrize(
    "size",
    [
        pytest.param(1, id="One_Value"),
        pytest.param(2, id='a  Symlink.leaving/the bundle'),
        pytest.param(3, id="already-kebab-3"),
        pytest.param(4, id=f"computed-{4}"),
    ],
)
def test_a_size_is_positive(size):
    assert size > 0
""",
            good="""import pytest


@pytest.mark.parametrize(
    "size",
    [
        pytest.param(1, id="one-value"),
        pytest.param(2, id='a-symlink-leaving-the-bundle'),
        pytest.param(3, id="already-kebab-3"),
        pytest.param(4, id=f"computed-{4}"),
    ],
)
def test_a_size_is_positive(size):
    assert size > 0
""",
        ),
        Example(
            bad="""import pytest


@pytest.mark.parametrize("size", [1, 2], ids=["Small (1)", "big_2"])
def test_a_size_is_positive(size):
    assert size > 0
""",
            good="""import pytest


@pytest.mark.parametrize("size", [1, 2], ids=["small-1", "big-2"])
def test_a_size_is_positive(size):
    assert size > 0
""",
        ),
    )
