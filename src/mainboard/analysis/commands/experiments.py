"""Experiment folders, their trial tests and the paper sections that cite them.

A research project here keeps each experiment as a folder with a `node.md` (the statement), a
`result.md` once settled and trial tests `test_*.py`; its papers cite an experiment by naming the
folder, in practice a `% source: experiments/<name>/result.md` comment above the text it supports.
"""

from pathlib import Path

from ..catalog import Relation
from .workspace import Workspace


def is_paper(path: str) -> bool:
    """Whether `path` is manuscript text: any LaTeX file, and Markdown beside a paper."""
    return path.endswith(".tex") or (path.endswith(".md") and "papers" in path.split("/")[:-1])


def read_papers(workspace: Workspace) -> None:
    """Load the manuscript text of the project into `paper_texts`, which `paper_mentions` reads."""
    workspace.read_texts("paper_texts", [path for path in workspace.files() if is_paper(path)])


def node_status(node: Path) -> str:
    """The `status:` of a node's front matter, empty without one."""
    try:
        head = node.read_text(encoding="utf-8").split("---")[1]
    except OSError, IndexError:
        return ""
    return next(
        (
            line.partition(":")[2].strip()
            for line in head.splitlines()
            if line.startswith("status:")
        ),
        "",
    )


class ExperimentNodes(Relation):
    """Each experiment folder: the topmost directory holding a `node.md`, with its settlement.

    settled: the folder holds a `result.md`.
    """

    stored = True
    query = r"""
        WITH nodes AS (
          SELECT regexp_extract(path, '^(.*)/node\.md$', 1) AS dir
          FROM main.ast_files
          WHERE language = 'markdown' AND path LIKE '%/node.md'
        )
        SELECT n.dir, regexp_extract(n.dir, '[^/]*$') AS name,
               EXISTS (SELECT 1 FROM main.ast_files f WHERE f.path = n.dir || '/result.md')
                 AS settled
        FROM nodes n
        WHERE NOT EXISTS (SELECT 1 FROM nodes o WHERE starts_with(n.dir, o.dir || '/'))
    """


class TrialFiles(Relation):
    """Each trial test (`test_*.py`, `*_test.py`) with the experiment folder it sits in, NULL when
    it sits in none."""

    stored = True
    query = r"""
        SELECT f.path, f.module, f.home, e.name AS experiment, e.dir AS experiment_dir
        FROM module_files() f
        LEFT JOIN experiment_nodes() e ON starts_with(f.path, e.dir || '/')
        WHERE regexp_matches(regexp_extract(f.path, '[^/]*$'), '^test_.*\.py$|_test\.py$')
    """


class PaperMentions(Relation):
    """The lines of `paper_texts` that cite an experiment folder, each with the heading above it.

    A line cites a folder by `experiments/<name>` or by `<name>/result.md` (or `node.md`,
    `section.md`).

    known: the name is an experiment folder; a name that is not stands for a renamed or removed
        experiment, or for a sibling directory such as `experiments/archive`.
    """

    stored = True
    query = r"""
        WITH lines AS (
          SELECT path, generate_subscripts(lines, 1) AS line, unnest(lines) AS text
          FROM (SELECT path, string_split(content, chr(10)) AS lines FROM paper_texts)
        ), cited AS (
          SELECT DISTINCT path, line, unnest(list_concat(
                   regexp_extract_all(text, 'experiments/([A-Za-z][A-Za-z0-9_]*)', 1),
                   regexp_extract_all(
                     text, '([a-z][a-z0-9]*(?:_[a-z0-9]+)+)/(?:result|node|section)\.md', 1)))
                 AS name
          FROM lines
        ), headings AS (
          SELECT path, line,
                 trim(CASE WHEN path LIKE '%.tex'
                           THEN regexp_extract(
                                  text, '\\(?:chapter|section|subsection)\*?\{(.*)\}', 1)
                           ELSE regexp_extract(text, '^#+\s+(.*)$', 1) END) AS title
          FROM lines
          WHERE regexp_matches(text, CASE WHEN path LIKE '%.tex'
                                          THEN '^\s*\\(chapter|section|subsection)\*?\{'
                                          ELSE '^#+\s' END)
        )
        SELECT c.path, c.line, c.name AS experiment, h.title AS section,
               e.name IS NOT NULL AS known
        FROM cited c
        LEFT JOIN experiment_nodes() e ON e.name = c.name
        ASOF LEFT JOIN headings h ON c.path = h.path AND c.line >= h.line
    """
