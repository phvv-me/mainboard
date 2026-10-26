"""ALL-TEST0005: helper modules of a test folder named outside its closed vocabulary."""

from .... import edits
from ....catalog import Example, Rule
from ....edits import Edit
from ....findings import Hit
from ..project.relocation import respelled
from .constants import ANCHOR

_IMPORTER = """\
from .footer import Footer
from .processes import ProcessLog


def test_a_footer_is_read_from_what_a_fake_process_printed():
    assert Footer and ProcessLog
"""
_IMPORTER_FIXED = _IMPORTER.replace(".footer", ".readers").replace(".processes", ".fakes")
_FIXTURE = """\
import pytest


@pytest.fixture
def lake():
    return 1
"""
_FOOTER = """\
from patos import FrozenModel

_UNITS = 1


class Footer(FrozenModel):
    rows: int

    @classmethod
    def of(cls, text: str) -> Footer:
        return cls(rows=int(text.split()[0]) * _UNITS)
"""
_PROCESS_LOG = """\
class ProcessLog:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
"""
_A_TEST = """\
def test_a_lake_verb_refuses_a_path_outside_the_bundle():
    pass
"""
_MIXED = """\
class Fake:
    pass


def make() -> Fake:
    return Fake()
"""
_FUNCTIONS = """\
def make() -> int:
    return 1
"""


class TestHelperVocabulary(Rule, id="ALL-TEST0005", applicability="review"):
    """Helper modules of a test folder named outside its closed vocabulary. A folder of tests
    holds the tests, and each helper under a name that says its role (`conftest` fixtures,
    `constants`, `fakes` doubles, `builders` of inputs, `readers` of what the code printed) so a
    stranger knows where to look and a `tools.py` never grows into a dumping ground.

    The vocabulary is a setting per language, empty for every one but Python, where the rule stays
    silent until a project configures a vocabulary. A module defining a test, and any folder
    under `exempt` (the harness), are skipped. The Python fix renames a module only when its
    public names say one role: classes that are all doubles are `fakes`, all parsed-output models
    are `readers`, functions are `builders`, assignments are `constants`, fixtures are
    `conftest`; a mixed module, classes of both kinds, or a taken name get no fix and the message
    says what blocks it. A rename respells every importer."""

    python: tuple[str, ...] = ("conftest", "constants", "fakes", "builders", "readers")
    typescript: tuple[str, ...] = ()
    swift: tuple[str, ...] = ()
    rust: tuple[str, ...] = ()
    exempt: tuple[str, ...] = ("gates",)

    query = r"""
        WITH languages AS (
          SELECT * FROM (VALUES
            ('python', '\.py$', '^(test_.*|.*_test)\.py$', $python::VARCHAR[]),
            ('typescript', '\.[cm]?[jt]sx?$', '\.(test|spec)\.[cm]?[jt]sx?$', $typescript::VARCHAR[]),
            ('swift', '\.swift$', 'Tests?\.swift$', $swift::VARCHAR[]),
            ('rust', '\.rs$', '(_test|^tests)\.rs$', $rust::VARCHAR[])
          ) AS v(language, source, tests, vocabulary)
          WHERE len(vocabulary) > 0
        ), entries AS (
          SELECT l.language, l.vocabulary, e.entry AS path,
                 regexp_extract(e.entry, '^((?:.*/)?)[^/]*$', 1) AS folder,
                 regexp_extract(e.entry, '(?:[^/]*)(\.[^./]+)$', 1) AS suffix,
                 regexp_replace(regexp_extract(e.entry, '[^/]+$'), l.source, '') AS stem,
                 regexp_matches(regexp_extract(e.entry, '[^/]+$'), l.tests) AS is_test_module
          FROM project_entries e JOIN languages l ON regexp_matches(e.entry, l.source)
          WHERE NOT e.is_directory
        ), holding AS (
          SELECT DISTINCT e.language, e.folder
          FROM entries e JOIN directory_facts d ON d.path = rtrim(e.folder, '/')
          WHERE e.is_test_module
        ), helpers AS (
          SELECT e.*, f.module
          FROM entries e
          JOIN holding h USING (language, folder)
          LEFT JOIN core_file f ON f.path = e.path
          WHERE NOT e.is_test_module AND e.stem <> '__init__'
            AND NOT list_contains(e.vocabulary, e.stem)
            AND NOT EXISTS (SELECT 1 FROM unnest(string_split(rtrim(e.folder, '/'), '/')) AS p(part)
                            WHERE list_contains($exempt::VARCHAR[], p.part))
            AND NOT EXISTS (SELECT 1 FROM core_def d
                            WHERE d.path = e.path AND d.owner_id IS NULL
                              AND (starts_with(d.name, 'test_') OR starts_with(d.name, 'Test')))
        ), shapes AS (
          SELECT h.path,
                 (SELECT count(*) FROM class_rows c
                  WHERE c.path = h.path AND NOT c.nested AND NOT starts_with(c.name, '_'))
                   AS classes,
                 (SELECT count(*) FROM class_rows c
                  WHERE c.path = h.path AND NOT c.nested AND NOT starts_with(c.name, '_')
                    AND (c.foundation_base OR c.is_declarative_model OR c.pydantic_base
                         OR c.inherits_approved_model_foundation
                         OR c.directly_inherits_pydantic_base_model)
                    AND EXISTS (SELECT 1 FROM class_method_defs m
                                WHERE m.class_id = c.class_id
                                  AND (m.kind = 'property'
                                       OR (m.kind IN ('class_method', 'static_method')
                                           AND (m.name IN ('of', 'parse', 'read', 'loads',
                                                           'from_text', 'from_output')
                                                OR starts_with(m.name, 'parse_')))))) AS readers,
                 (SELECT count(*) FROM class_rows c
                  WHERE c.path = h.path AND NOT c.nested AND NOT starts_with(c.name, '_')
                    AND NOT (c.foundation_base OR c.is_declarative_model OR c.pydantic_base
                             OR c.inherits_approved_model_foundation
                             OR c.directly_inherits_pydantic_base_model)
                    AND NOT c.is_protocol AND NOT c.is_dataclass
                    AND NOT EXISTS (SELECT 1 FROM class_method_defs m
                                    WHERE m.class_id = c.class_id
                                      AND m.kind IN ('class_method', 'static_method')
                                      AND (m.name IN ('of', 'parse', 'read', 'loads',
                                                      'from_text', 'from_output')
                                           OR starts_with(m.name, 'parse_')))
                    AND NOT EXISTS (SELECT 1 FROM class_bases b
                                    WHERE b.class_id = c.class_id
                                      AND regexp_matches(b.base_name,
                                            '(Protocol|Enum|Flag|TypedDict|NamedTuple|Exception|Error)$')))
                   AS doubles,
                 (SELECT count(*) FROM core_def d
                  WHERE d.path = h.path AND d.owner_id IS NULL AND d.kind = 'function'
                    AND d.type = 'function_definition' AND NOT starts_with(d.name, '_')
                    AND NOT EXISTS (SELECT 1 FROM core_decor x
                                    WHERE x.file_id = d.file_id AND x.def_id = d.def_id
                                      AND x.name IN ('pytest.fixture', 'fixture',
                                                     'pytest_asyncio.fixture'))) AS functions,
                 (SELECT count(*) FROM core_def d
                  JOIN core_decor x ON x.file_id = d.file_id AND x.def_id = d.def_id
                  WHERE d.path = h.path AND d.owner_id IS NULL AND d.kind = 'function'
                    AND d.type = 'function_definition' AND NOT starts_with(d.name, '_')
                    AND x.name IN ('pytest.fixture', 'fixture', 'pytest_asyncio.fixture'))
                   AS fixtures,
                 (SELECT count(*) FROM ast_nodes e
                  JOIN ast_nodes a ON a.fact_id = e.fact_id AND a.parent_id = e.node_id
                                  AND a.type = 'assignment'
                  WHERE e.path = h.path AND e.depth = 1 AND e.type = 'expression_statement'
                    AND NOT starts_with(a.name, '_'))
                 + (SELECT count(*) FROM ast_nodes t
                    WHERE t.path = h.path AND t.depth = 1 AND t.type = 'type_alias_statement')
                   AS assignments,
                 (SELECT count(*) FROM ast_nodes n
                  WHERE n.path = h.path AND n.depth = 1
                    AND n.type NOT IN ('import_statement', 'import_from_statement',
                                       'future_import_statement', 'function_definition',
                                       'class_definition', 'decorated_definition', 'comment',
                                       'pass_statement', 'type_alias_statement',
                                       'expression_statement'))
                 + (SELECT count(*) FROM ast_nodes e
                    WHERE e.path = h.path AND e.depth = 1 AND e.type = 'expression_statement'
                      AND NOT EXISTS (SELECT 1 FROM ast_nodes a
                                      WHERE a.fact_id = e.fact_id AND a.parent_id = e.node_id
                                        AND a.type IN ('assignment', 'string'))) AS others
          FROM helpers h WHERE h.language = 'python'
        ), roles AS (
          SELECT h.*, s.*,
                 CASE WHEN s.others > 0 THEN NULL
                      WHEN s.classes > 0 AND s.functions + s.fixtures + s.assignments = 0
                      THEN CASE WHEN s.readers = s.classes THEN 'readers'
                                WHEN s.doubles = s.classes THEN 'fakes' END
                      WHEN s.functions > 0 AND s.classes + s.fixtures + s.assignments = 0
                      THEN 'builders'
                      WHEN s.assignments > 0 AND s.classes + s.functions + s.fixtures = 0
                      THEN 'constants'
                      WHEN s.fixtures > 0 AND s.classes + s.functions + s.assignments = 0
                      THEN 'conftest' END AS role
          FROM helpers h JOIN shapes s USING (path)
        ), classified AS (
          SELECT r.*, row_number() OVER (PARTITION BY r.folder, r.role ORDER BY r.path) AS place
          FROM roles r
        ), planned AS (
          SELECT c.*,
                 CASE WHEN c.role IS NOT NULL AND list_contains(c.vocabulary, c.role)
                           AND c.place = 1
                           AND NOT EXISTS (SELECT 1 FROM entries x
                                           WHERE x.language = c.language AND x.folder = c.folder
                                             AND x.stem = c.role)
                      THEN c.role END AS target,
                 CASE
                   WHEN c.others > 0 THEN 'it runs statements besides defining names'
                   WHEN c.classes + c.functions + c.fixtures + c.assignments = 0
                   THEN 'it defines no public name to classify it by'
                   WHEN CAST(c.classes > 0 AS INTEGER) + CAST(c.functions > 0 AS INTEGER)
                        + CAST(c.fixtures > 0 AS INTEGER) + CAST(c.assignments > 0 AS INTEGER) > 1
                   THEN 'it mixes ' || array_to_string(list_filter([
                          CASE WHEN c.classes > 0 THEN 'classes' END,
                          CASE WHEN c.functions > 0 THEN 'functions' END,
                          CASE WHEN c.fixtures > 0 THEN 'fixtures' END,
                          CASE WHEN c.assignments > 0 THEN 'assignments' END],
                          lambda kind: kind IS NOT NULL), ' and ')
                   WHEN c.role IS NULL
                   THEN 'its classes are not all doubles or all parsed-output models'
                   WHEN NOT list_contains(c.vocabulary, c.role)
                   THEN 'the vocabulary has no `' || c.role || '`'
                   WHEN c.place > 1
                   THEN 'another module of the folder takes `' || c.role || '` first'
                   ELSE '`' || c.role || c.suffix || '` already exists in the folder'
                 END AS blocker
          FROM classified c
        ), affected AS (
          SELECT p.path AS moved, f.path, f.module, f.is_package, s.content
          FROM planned p
          JOIN core_file f ON f.language = 'python' AND f.path <> p.path
          JOIN source_files s ON s.path = f.path
          WHERE p.target IS NOT NULL
            AND EXISTS (SELECT 1 FROM core_import i
                        WHERE i.file_id = f.file_id
                          AND (i.source_module = p.module OR i.target = p.module))
        )
        SELECT h.path, 1 AS start_line, 0 AS start_column, h.language, h.folder, h.module,
               h.target, h.suffix,
               format('`{}` sits among tests but is named outside {}; {}', h.path,
                      array_to_string(h.vocabulary, ', '),
                      CASE WHEN h.language <> 'python' THEN 'name it from that vocabulary'
                           WHEN h.target IS NOT NULL
                           THEN format('it holds one role, so rename it `{}`',
                                       h.folder || h.target || h.suffix)
                           ELSE 'no rename: ' || h.blocker END) AS message,
               list({'path': a.path, 'module': a.module, 'package': a.is_package,
                     'content': a.content}) FILTER (WHERE a.path IS NOT NULL) AS files
        FROM (SELECT e.path, e.language, e.folder, e.vocabulary, e.suffix, p.module, p.target,
                     p.blocker
              FROM helpers e LEFT JOIN planned p USING (path)) h
        LEFT JOIN affected a ON a.moved = h.path
        GROUP BY ALL
    """
    examples = (
        Example(
            bad=_IMPORTER,
            good=_IMPORTER_FIXED,
            path="tests/lake/test_core.py",
            context={
                "tests/lake/__init__.py": "",
                "tests/lake/constants.py": "MAPS = 1\n",
                "tests/lake/conftest.py": _FIXTURE,
                "tests/lake/footer.py": _FOOTER,
                "tests/lake/processes.py": _PROCESS_LOG,
                "tests/lake/runner.py": _A_TEST,
                "tests/gates/harness.py": "HARNESS = 1\n",
                "tests/gates/test_harness.py": _A_TEST,
            },
        ),
        Example(
            bad=_MIXED,
            path="tests/lake/tools.py",
            context={"tests/lake/test_core.py": _A_TEST},
        ),
        Example(
            bad=_FUNCTIONS,
            path="tests/lake/helpers.py",
            context={
                "tests/lake/test_core.py": _A_TEST,
                "tests/lake/builders.py": "def build() -> int:\n    return 2\n",
            },
        ),
        Example(
            bad="export const util = 1;\n",
            path="web/util.ts",
            context={
                **ANCHOR,
                "pyproject.toml": '[tool.mcmr.rules."ALL-TEST0005"]\ntypescript = ["fakes"]\n',
                "web/a.test.ts": 'it("works", () => {});\n',
                "web/fakes.ts": "export const fake = 1;\n",
            },
        ),
    )

    def repair(self, hit: Hit) -> list[Edit] | None:
        """Rename the module, then respell the imports of it."""
        if hit.target is None:
            return None
        module = str(hit.module)
        sibling = ".".join([*module.split(".")[:-1], str(hit.target)])
        made = respelled(hit, {module: sibling})
        if made is None:
            return None
        return [*made, edits.rename(str(hit.path), target=f"{hit.folder}{hit.target}{hit.suffix}")]
