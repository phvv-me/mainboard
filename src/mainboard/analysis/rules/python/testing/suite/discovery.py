"""Rules over how pytest finds and isolates tests (mcmr's PY-TEST discovery group)."""

from .....catalog import Example, Rule
from ...constants.testing_suite import PYTEST_TABLE
from .core import pytest_option_fix


class ConftestImport(Rule, id="PY-TEST0001"):
    """Imports of `conftest`, which pytest loads on its own."""

    allowed_modules: tuple[str, ...] = ()

    query = """
        SELECT path, start_line, start_column, 'import:' || path || ':' || name AS fact_id
        FROM core_import
        WHERE (name = 'conftest' OR imported_name = 'conftest'
               OR list_contains(str_split(module, '.'), 'conftest'))
          AND NOT list_contains($allowed_modules::VARCHAR[], module)
    """
    message = "`{fact_id}` satisfies conftest import"


class LegacyTmpdirFixtureCount(Rule, id="PY-TEST0002"):
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
