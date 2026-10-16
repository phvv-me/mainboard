"""What the PY-CUDA facts know about names: the definitions of a project, what each import binds,
and where a bound name finally lands, through re-exports, so a type written `Vector[Index]`
resolves to `patos.cuda.scalars.Vector` however many modules hand it on.

Every relation here is stored: computed once per fact base from the syntax trees (`cuda_nodes()`) and
the verified import facts.
"""

from .cuda_base import GPU_MODULE_PATTERN, CudaRelation
from .imports_common import MODULE_NAMES


class CudaModules(CudaRelation):
    """The dotted module name of every Python file."""

    stored = True
    query = f"""
        WITH {MODULE_NAMES}
        SELECT fact_id, path, module, regexp_matches(path, '(^|/)__init__\\.pyi?$') AS is_package
        FROM module_names
    """


class CudaGpuFiles(CudaRelation):
    """The files that import a GPU framework (CuPy, torch, numba, `cuda`, Triton, Helion, Warp,
    JAX, TileLang, patos's GPU packages)."""

    stored = True
    query = f"""
        SELECT DISTINCT fact_id FROM cuda_imports()
        WHERE regexp_matches(module, '{GPU_MODULE_PATTERN}')
    """


class CudaNodes(CudaRelation):
    """The syntax nodes of a project that has any GPU file, and none otherwise: every other fact
    reads the trees through this, so a project with no GPU code pays almost nothing."""

    query = """
        SELECT * FROM ast_nodes WHERE EXISTS (SELECT 1 FROM cuda_gpu_files())
    """


class CudaDefinitions(CudaRelation):
    """Every class, function and module-level variable a file defines: its qualified name
    (`Kernel.launch`), the key `path::qualname` other relations refer to it by, and its dotted
    name from outside (`patos.cuda.typed.kernels.Kernel.launch`). A method names its class
    (`class_id`), a nested function its enclosing function (`outer_id`)."""

    stored = True
    query = r"""
        WITH defined AS (
          SELECT n.fact_id, n.path, n.node_id,
                 CASE n.type WHEN 'class_definition' THEN 'class'
                             WHEN 'function_definition' THEN 'function'
                             ELSE 'variable' END AS kind, n.name,
                 CASE WHEN n.type = 'assignment' THEN n.name
                      ELSE array_to_string(list_transform(n.qualified_name, lambda q: q.name), '.')
                 END AS qualname,
                 n.scope."class" AS class_id, n.scope."function" AS outer_id,
                 n.start_line, n.start_column, n.end_line, n.end_column
          FROM cuda_nodes() n
          WHERE n.type IN ('class_definition', 'function_definition')
             OR (n.type = 'assignment' AND n.scope."function" IS NULL AND n.scope."class" IS NULL
                 AND regexp_matches(n.name, '^[A-Za-z_][A-Za-z0-9_]*$'))
        )
        SELECT d.*, d.path || '::' || d.qualname AS key, m.module || '.' || d.qualname AS qualified
        FROM defined d JOIN cuda_modules() m ON m.fact_id = d.fact_id
    """


class CudaImports(CudaRelation):
    """What each import binds, as an absolute dotted name: `cp` -> `cupy`, `Vector` ->
    `cutoken.types.Vector` (a relative import is made absolute from its importer's module)."""

    stored = True
    query = """
        SELECT m.fact_id, b.path, b.name AS bound, b.is_type_only, b.start_line, b.start_column,
               b.end_line, b.end_column,
               CASE WHEN b.relative_level = 0 THEN b.module
                    ELSE concat_ws('.', nullif(array_to_string(
                           string_split(b.importer_module, '.')[1:greatest(0,
                             len(string_split(b.importer_module, '.')) - b.relative_level
                             + CASE WHEN m.is_package THEN 1 ELSE 0 END)], '.'), ''),
                         nullif(b.module, '')) END AS module,
               b.imported_name
        FROM import_binding_facts b JOIN cuda_modules() m ON m.path = b.path
        WHERE NOT b.is_wildcard
    """


class CudaNames(CudaRelation):
    """Where a name a file binds by import finally lands: `qualified` is the dotted name outside
    the project (`patos.cuda.scalars.Vector`) or, when a project file defines it, `definition`
    the key of that definition (a re-export is followed, up to four modules deep). `seen` lists
    every dotted name the name had on the way, the one the import wrote first
    (`patos.cuda.typed.device`) and each re-export behind it."""

    stored = True
    query = """
        WITH RECURSIVE reach(fact_id, bound, module, imported, definition, hops, seen)
          AS (
          SELECT fact_id, bound, module, imported_name, NULL::VARCHAR, 0,
                 [CASE WHEN module = imported_name OR imported_name = '' THEN module
                       ELSE module || '.' || imported_name END]
          FROM cuda_imports()
          UNION ALL
          SELECT r.fact_id, r.bound, coalesce(s.module, r.module),
                 coalesce(s.imported_name, r.imported),
                 CASE WHEN s.fact_id IS NULL THEN d.key END, r.hops + 1,
                 list_append(r.seen,
                   CASE WHEN coalesce(s.module, r.module) = coalesce(s.imported_name, r.imported)
                          OR coalesce(s.imported_name, r.imported) = ''
                        THEN coalesce(s.module, r.module)
                        ELSE coalesce(s.module, r.module) || '.'
                             || coalesce(s.imported_name, r.imported) END)
          FROM reach r
          JOIN cuda_modules() m ON m.module = r.module
          LEFT JOIN cuda_imports() s ON s.fact_id = m.fact_id AND s.bound = r.imported
          LEFT JOIN cuda_definitions() d ON d.fact_id = m.fact_id AND d.name = r.imported
            AND d.kind <> 'variable' AND d.qualname = d.name
          WHERE r.hops < 4 AND r.definition IS NULL AND (s.fact_id IS NOT NULL OR d.key IS NOT NULL)
        )
        SELECT DISTINCT ON (fact_id, bound) fact_id, bound, definition, seen,
               CASE WHEN module = imported OR imported = '' THEN module
                    ELSE module || '.' || imported END AS qualified
        FROM reach
        ORDER BY fact_id, bound, hops DESC
    """
