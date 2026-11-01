"""`module_readers` is the `files` row `move-modules` respells: the modules a move carries and
every module that imports one or reads it as `a.b.NAME` through a plain `import a`."""

from pathlib import Path

from mainboard.analysis.engine.lint import Lint, load_catalog
from mainboard.analysis.rules.general.dependencies.modules import ModuleInception
from mainboard.analysis.rules.python.design.importers import ModuleReaders

from ....engine.builders import written

_MODULE = "pkg.rules.rules"
_PROJECT = {
    "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
    "src/pkg/__init__.py": "",
    "src/pkg/rules/__init__.py": "",
    "src/pkg/rules/other.py": "Q = 1\n",
    "src/pkg/rules/rules.py": "X = 1\n",
}
_READERS = {
    "src/pkg/absolute.py": "from pkg.rules.rules import X\n",
    "src/pkg/relative.py": "from .rules.rules import X\n",
    "src/pkg/member.py": "from pkg.rules import rules\n",
    "src/pkg/plain.py": "import pkg.rules.rules\n",
    "src/pkg/aliased.py": "import pkg.rules.rules as r\n",
    "src/pkg/dotted.py": "import pkg.rules\n\nY = pkg.rules.rules.X\n",
}
_STRANGERS = {
    "src/pkg/sibling.py": "from pkg.rules.other import Q\n",
    "src/pkg/package.py": "from pkg.rules import other\n",
    "src/pkg/reads_a_sibling.py": "import pkg.rules\n\nY = pkg.rules.other.Q\n",
    "src/pkg/names_it.py": "# pkg.rules.rules\nY = 'pkg.rules.rules'\n",
    "notes.md": "pkg.rules.rules\n",
}


def test_the_readers_of_a_module_are_the_module_and_the_files_that_import_or_read_it(
    tmp_path: Path,
) -> None:
    assert ModuleReaders.name() == "module_readers"
    root = written(tmp_path, {**_PROJECT, **_READERS, **_STRANGERS})
    load_catalog()

    with Lint(root, (ModuleInception,)) as lint:
        cursor = lint.facts(root).cursor(None)
        files = cursor.execute("SELECT files FROM module_readers([?])", [_MODULE]).fetchone()

    assert files is not None
    assert [file["path"] for file in files[0]] == sorted(["src/pkg/rules/rules.py", *_READERS])
