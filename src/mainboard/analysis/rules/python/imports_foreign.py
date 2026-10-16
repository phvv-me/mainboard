"""Rules over what a project imports from the distributions it does not own."""

import sys

from pathspec import PathSpec
from pathspec.pattern import RegexPattern

from ...catalog import Example, Rule
from ...facts import Setting
from .cuda_paths import MEASURING

_INTERNAL = (
    r"^numba\.core\.(typing|imputils|cgutils|compiler|lowering|datamodel|types|dispatcher|codegen)"
    r"(\.|$)",
    r"^numba\.cuda\.(cudadecl|cudaimpl|codegen|target|typing|core|dispatcher|cudadrv|cgutils"
    r"|compiler|lowering)(\.|$)",
)


class ForeignPrivateApi(Rule, id="PY-IMPO0007", applicability="none"):
    """Imports of the private or internal parts of a distribution the project does not own
    (`cuda.core._utils`, `torch._dynamo`, `numba.cuda.cudaimpl`) and assignments onto attributes
    of what such a distribution provides (monkeypatches), outside the files a project fences off
    to hold them; the standard library is no distribution here. A module or name with a
    `_`-prefixed segment is private, and so is a module
    that matches an `internal` pattern: the parts a distribution leaves unprefixed but documents
    as internal. A project's own modules are PY-IMPO0002's and PY-IMPO0005's; a patch that a
    `finally` undoes is not reported."""

    exempt: tuple[str, ...] = (*MEASURING, r"\.pyi$")
    fence: tuple[str, ...] = ()
    internal: tuple[str, ...] = _INTERNAL

    query = r"""
        WITH own AS (
          SELECT DISTINCT split_part(importer_module, '.', 1) AS root FROM import_binding_facts
        ), foreign_imports AS (
          SELECT b.*, split_part(b.module, '.', 1) AS distribution
          FROM import_binding_facts b
          WHERE b.relative_level = 0 AND NOT b.is_wildcard
            AND split_part(b.module, '.', 1) NOT IN (SELECT root FROM own)
            AND NOT list_contains($stdlib, split_part(b.module, '.', 1))
        ), imported AS (
          SELECT i.path, i.start_line, i.start_column, i.end_line, i.end_column, i.distribution,
                 i.module || CASE WHEN i.imported_name = i.module THEN ''
                                  ELSE '.' || i.imported_name END AS imported
          FROM foreign_imports i
          WHERE regexp_matches(i.module, '(^|\.)_[A-Za-z]')
             OR regexp_matches(i.imported_name, '^_[A-Za-z]')
             OR coalesce(list_bool_or(list_transform($internal,
                  lambda p: regexp_matches(i.module || '.' || i.imported_name, p))), false)
        ), grouped AS (
          SELECT path, distribution, min(start_line) AS start_line,
                 min(start_column) AS start_column, min(end_line) AS end_line,
                 min(end_column) AS end_column, 'import' AS problem,
                 count(DISTINCT imported) AS n,
                 array_to_string(list_sort(list(DISTINCT '`' || imported || '`')), ', ') AS names
          FROM imported GROUP BY path, distribution
        ), patched AS (
          SELECT a.path, g.distribution, a.start_line, a.start_column, a.end_line, a.end_column,
                 'patch' AS problem, 1 AS n, '`' || l.name || '`' AS names
          FROM ast_nodes a
          JOIN ast_nodes l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
            AND l.sibling_index = 0 AND l.type = 'attribute'
          JOIN foreign_imports g ON g.path = a.path AND g.name = split_part(l.name, '.', 1)
          WHERE a.type = 'assignment'
            AND EXISTS (
              SELECT 1 FROM ast_nodes v
              WHERE v.fact_id = a.fact_id AND v.parent_id = a.node_id
                AND v.sibling_index = (SELECT max(w.sibling_index) FROM ast_nodes w
                                       WHERE w.fact_id = a.fact_id AND w.parent_id = a.node_id)
                AND (v.type IN ('lambda', 'call')
                     OR (v.type IN ('identifier', 'attribute') AND (
                           EXISTS (SELECT 1 FROM ast_nodes d
                                   WHERE d.fact_id = v.fact_id AND d.name = v.name
                                     AND d.type IN ('function_definition', 'class_definition'))
                           OR EXISTS (SELECT 1 FROM import_binding_facts o
                                      WHERE o.path = a.path AND o.name = split_part(v.name, '.', 1)
                                        AND (o.relative_level > 0
                                             OR split_part(o.module, '.', 1)
                                                IN (SELECT root FROM own)))))))
            AND NOT EXISTS (SELECT 1 FROM ast_nodes undo
                            WHERE undo.fact_id = a.fact_id AND undo.type = 'finally_clause'
                              AND a.node_id BETWEEN undo.node_id + 1
                                  AND undo.node_id + undo.descendant_count)
            AND NOT EXISTS (
              SELECT 1 FROM ast_nodes fin
              JOIN ast_nodes b ON b.fact_id = fin.fact_id AND b.type = 'assignment'
                AND b.node_id BETWEEN fin.node_id + 1 AND fin.node_id + fin.descendant_count
              JOIN ast_nodes bl ON bl.fact_id = b.fact_id AND bl.parent_id = b.node_id
                AND bl.sibling_index = 0 AND bl.name = l.name
              WHERE fin.fact_id = a.fact_id AND fin.type = 'finally_clause'
                AND b.node_id <> a.node_id)
        )
        SELECT f.*,
               CASE f.problem
                 WHEN 'import' THEN 'imports ' || f.names || ' from ' || f.distribution
                 ELSE 'assigns onto ' || f.names || ' of ' || f.distribution || ' and leaves it so'
               END AS what,
               CASE f.problem
                 WHEN 'import' THEN 'these are private or internal and change between releases '
                      || 'without notice, so an upgrade breaks this file at import'
                 ELSE 'the patch is seen by every user of the distribution in the process, and '
                      || 'breaks when what it replaces changes' END AS cost
        FROM (SELECT * FROM grouped UNION ALL SELECT * FROM patched) f
        WHERE NOT coalesce(list_bool_or(list_transform($exempt,
                lambda p: regexp_matches(f.path, p))), false)
          AND NOT coalesce(list_bool_or(list_transform($fence_patterns,
                lambda p: regexp_matches(f.path, p))), false)
    """
    message = (
        "This file {what}: {cost}. Fix: keep it in one adapter package with a test that imports "
        "it, and use the public API elsewhere. To allow that package, list its files in the "
        'project\'s `pyproject.toml`: `[tool.mcmr.rules."PY-IMPO0007"]` with '
        '`fence = ["src/app/adapter/**"]` (gitignore-style globs matched against file paths)'
    )
    examples = (
        Example(
            bad="from cuda.core._utils import cuda_utils\n\nERROR = cuda_utils.CUDAError\n",
            good="from cuda.core import Device\n\nDEVICE = Device\n",
        ),
        Example(
            bad="from numba.cuda import codegen\n\n\ndef patch():\n"
            "    codegen.JITCUDACodegen._library_class = object\n",
            good="def patch():\n    return None\n",
        ),
        Example(
            bad="from torch import _dynamo\n\nCONFIG = _dynamo.config\n",
            good="import torch\n\nCOMPILE = torch.compile\n",
        ),
        Example(
            bad="import requests\n\n\ndef patch():\n    requests.Session.send = lambda self: None\n",
            good="import requests\n\n\ndef patch():\n    return requests.Session()\n",
        ),
    )

    def parameters(self) -> dict[str, Setting]:
        spec = PathSpec.from_lines("gitwildmatch", self.fence)
        return super().parameters() | {
            "fence_patterns": [
                pattern.regex.pattern
                for pattern in spec.patterns
                if isinstance(pattern, RegexPattern) and pattern.regex
            ],
            "stdlib": sorted(sys.stdlib_module_names),
        }
