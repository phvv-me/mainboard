"""mcmr's InteropFact, derived from the sources' text and sitting_duck's TOML trees.

An artifact is something one language declares for another to reach: a `[[bin]]` of a Cargo
manifest (binary), an entry of `[project.scripts]` (console script), a `#[pymodule]` function
(native module, Rust) or `PYBIND11_MODULE` (native module, C++), a `__global__` function (kernel,
CUDA), a library a Python file loads with `CDLL(` or `cdll.LoadLibrary(` (shared library, named by
the argument's text up to the first `)`). One fact per name and mechanism, declared in the first
file (in path order) that states it, with every other interesting file that names it between
quotes as a reference (its first such line). Differences kept on purpose:

- npm `bin` entries of `package.json` are not read: the file is not a parsed language, so its text
  is not among the sources;
- a CUDA kernel's name is the word before the parenthesis that opens its parameters, after
  `__launch_bounds__(...)` and comments are dropped, where the kernel read the declaration
  character by character;
- files are taken in path order where the kernel took the directory walk's order.
"""

from ....catalog import Family

_MECHANISMS = "'binary', 'console-script', 'native-module', 'shared-library', 'kernel'"

_SOURCES = r"""
    sources AS (
      SELECT s.path, s.content, split_part(s.content, '#[cfg(test)]', 1) AS declared_text,
             CASE WHEN regexp_matches(s.path, '\.rs$') THEN 'rust'
                  WHEN regexp_matches(s.path, '\.(cu|cuh)$') THEN 'cuda'
                  WHEN regexp_matches(s.path, '\.(cpp|cc|hpp)$') THEN 'cpp'
                  WHEN regexp_matches(s.path, '\.h$') THEN 'c'
                  WHEN regexp_matches(s.path, '\.py$') THEN 'python'
                  WHEN regexp_matches(s.path, '\.(ts|tsx)$') THEN 'typescript'
                  ELSE 'manifest' END AS language
      FROM source_files s
      WHERE regexp_matches(s.path, '(Cargo\.toml|package\.json|pyproject\.toml)$')
         OR regexp_matches(s.path, '\.(rs|cpp|cc|cu|cuh|h|hpp|py|ts|tsx)$')
    )"""

_FACTS = (
    "WITH "
    + _SOURCES
    + r""", manifests AS (
      SELECT s.path, trim(CASE WHEN a.name = 'bin' THEN v.name ELSE p.name END, '"''') AS name,
             CASE WHEN a.name = 'bin' THEN 'binary' ELSE 'console-script' END AS mechanism,
             CASE WHEN a.name = 'bin' THEN 'rust' ELSE 'python' END AS language
      FROM ast_nodes a
      JOIN ast_files f ON f.fact_id = a.fact_id
      JOIN sources s ON s.path = f.path
      JOIN ast_nodes p ON p.fact_id = a.fact_id AND p.parent_id = a.node_id AND p.type = 'pair'
      LEFT JOIN ast_nodes v ON v.fact_id = a.fact_id AND v.parent_id = p.node_id
                           AND v.sibling_index = 2 AND v.type = 'string'
      WHERE f.language = 'toml'
        AND ((a.type = 'table_array_element' AND a.name = 'bin' AND s.path LIKE '%Cargo.toml'
              AND p.name = 'name' AND v.node_id IS NOT NULL)
             OR (a.type = 'table' AND a.name = 'project.scripts'
                 AND s.path LIKE '%pyproject.toml' AND v.node_id IS NOT NULL))
    ), pymodules AS (
      SELECT path, regexp_extract(substr(part, strpos(part, 'fn ') + 3), '^\s*(\w+)', 1) AS name,
             'native-module' AS mechanism, 'rust' AS language
      FROM (SELECT path, unnest(string_split(declared_text, '#[pymodule]')[2:]) AS part
            FROM sources WHERE language = 'rust')
      WHERE strpos(part, 'fn ') > 0
    ), pybind AS (
      SELECT path, trim(trim(unnest(regexp_extract_all(declared_text,
                                                       'PYBIND11_MODULE\(([^,]*),', 1))), '"'),
             'native-module', 'cpp'
      FROM sources WHERE regexp_matches(path, '\.(cpp|cc|cu)$')
    ), libraries AS (
      SELECT path, trim(trim(unnest(regexp_extract_all(
                                      declared_text,
                                      '(?:CDLL|cdll\.LoadLibrary)\(([^)]*)\)', 1))), '"'),
             'shared-library', 'c'
      FROM sources WHERE language = 'python'
    ), kernels AS (
      SELECT path, unnest(regexp_extract_all(
               regexp_replace(regexp_replace(regexp_replace(content, '/\*(?s:.*?)\*/', '', 'g'),
                                             '//[^\n]*', '', 'g'),
                              '__launch_bounds__\s*\([^)]*\)', '', 'g'),
               '__global__[^;{}()]*?\b(\w+)\s*\(', 1)) AS name,
             'kernel' AS mechanism, 'cuda' AS language
      FROM sources WHERE language = 'cuda'
    ), declarations AS (
      SELECT * FROM manifests
      UNION ALL SELECT * FROM pymodules
      UNION ALL SELECT * FROM pybind
      UNION ALL SELECT * FROM libraries
      UNION ALL SELECT * FROM kernels
    ), named AS (
      SELECT name, mechanism, arg_min(path, path) AS declared_in,
             arg_min(language, path) AS declared_language
      FROM declarations WHERE name <> '' GROUP BY name, mechanism
    )
    SELECT row_number() OVER (ORDER BY name, list_position(["""
    + _MECHANISMS
    + r"""], mechanism)) - 1
             AS fact_order,
           'interop:' || mechanism || ':' || name AS fact_id, declared_in AS path,
           1 AS start_line, 0 AS start_column, 1 AS end_line, 0 AS end_column,
           declared_language AS language, declared_language, mechanism, name
    FROM named
"""
)

_REFERENCES = (
    "WITH "
    + _SOURCES
    + """
    SELECT f.fact_id, row_number() OVER (PARTITION BY f.fact_id ORDER BY s.path) - 1 AS ordinal,
           s.path, s.language,
           list_position(list_transform(string_split(s.content, chr(10)),
             lambda l: contains(l, '"' || f.name || '"') OR contains(l, '''' || f.name || '''')
                  OR contains(l, '`' || f.name || '`')), true) AS line
    FROM interop_facts f
    JOIN sources s ON s.path <> f.path
                  AND (contains(s.content, '"' || f.name || '"')
                       OR contains(s.content, '''' || f.name || '''')
                       OR contains(s.content, '`' || f.name || '`'))
"""
)


class Interop(Family):
    family = "InteropFact"
    relations = {"facts": _FACTS, "references": _REFERENCES}
