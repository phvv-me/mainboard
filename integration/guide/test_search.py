"""A search reads the README block by block: prose whole, code and table lines one by one."""

from pathlib import Path

import pytest
from cyclopts import App

from mainboard.guide.search import Help

_README = """# Widgets

Widgets spin.
They also hum.

```console
$ mb widget --size 3
```

| name | note |
|------|------|
| gadget | blue |

## Parts

- gear
- belt
"""


@pytest.mark.parametrize(
    ("query", "name", "line", "excerpt"),
    [
        pytest.param("hum", "docs Widgets", 3, "They also hum.", id="a-prose-block-whole"),
        pytest.param("size", "docs Widgets", 7, "$ mb widget --size 3", id="a-code-line-alone"),
        pytest.param("gadget", "docs Widgets", 12, "| gadget | blue |", id="a-table-row-alone"),
        pytest.param("belt", "docs Parts", 16, "- belt", id="a-list-under-its-heading"),
        pytest.param("# widgets", "docs Widgets", 1, "# Widgets", id="a-heading-line-itself"),
    ],
)
def test_a_piece_of_the_readme_is_found_under_the_heading_above_it(
    tmp_path: Path, query: str, name: str, line: int, excerpt: str
) -> None:
    (tmp_path / "README.md").write_text(_README, encoding="utf-8")
    searching = Help(App())
    searching.package = tmp_path
    assert searching.show(query) == f"{name}\n  {tmp_path / 'README.md'}:{line}\n  {excerpt}\n\n"
