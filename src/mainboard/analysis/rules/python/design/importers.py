"""Who reads a project module's top-level names: the relations a move of definitions rewrites."""

from ....catalog import Function, Relation

_IMPORTED = """
        SELECT coalesce(i.source_module, i.module) AS module,
               if(i.imported_name <> '', i.imported_name, i.name) AS name,
               i.path, i.start_line, i.start_column, i.end_line, i.end_column,
               f.module AS importer,
               if(is_initializer_path(i.path), string_split(f.module, '.'),
                  list_slice(string_split(f.module, '.'), 1, -2)) AS importer_package,
               array_to_string(
                 list_slice(string_split(s.content, chr(10)), i.start_line, i.end_line),
                 chr(10)) AS statement
        FROM core_import i
        JOIN core_file f ON f.file_id = i.file_id
        JOIN source_files s ON s.path = i.path
        WHERE i.statement_type = 'import_from_statement' AND {owned}
    """
_REACHED = """
        SELECT i.target AS module, substr(n.name, strlen(i.name) + 2) AS name, f.path,
               n.start_line, n.start_column, n.end_line, n.end_column,
               f.module AS importer,
               if(is_initializer_path(f.path), string_split(f.module, '.'),
                  list_slice(string_split(f.module, '.'), 1, -2)) AS importer_package,
               i.name AS binding, i.start_line AS statement_start_line,
               i.start_column AS statement_start_column, i.end_line AS statement_end_line,
               i.end_column AS statement_end_column,
               array_to_string(
                 list_slice(string_split(s.content, chr(10)), i.start_line, i.end_line),
                 chr(10)) AS statement,
               s.content
        FROM core_import i
        JOIN core_file f ON f.file_id = i.file_id AND f.module <> i.target
        JOIN core_node n ON n.file_id = i.file_id AND n.type = 'attribute'
          AND starts_with(n.name, i.name || '.')
          AND NOT contains(substr(n.name, strlen(i.name) + 2), '.')
        JOIN source_files s ON s.path = f.path
        WHERE {owned}
    """


class ModulePath(Function):
    """The path of the module a definition gets in `folder` when it moves into one of its own:
    `HTTPServer` is `folder/http_server.py`, and a root folder (empty) gives `http_server.py`."""

    folder: str
    name: str

    expression = r"""coalesce(nullif(folder, '') || '/', '')
        || lower(regexp_replace(regexp_replace(name, '([a-z0-9])([A-Z])', '\1_\2', 'g'),
                                '([A-Z]+)([A-Z][a-z])', '\1_\2', 'g')) || '.py'"""


class ImportedNames(Relation):
    """Each name a project `from ... import` statement takes from a project module, with the
    importing module (`importer`), the package it imports from (`importer_package`, the module
    itself for an `__init__.py`) and the statement's own lines (`statement`). The module is the
    importer's own top package or reached by a relative import."""

    query = _IMPORTED.format(owned="i.is_project_owned")


class ProjectImportedNames(Relation):
    """`imported_names()` for every importer, wherever its file stands: a module is the
    project's when a file of it stands in the project, so `integration/` reading `mainboard.*` by
    its full name is an importer too."""

    query = _IMPORTED.format(
        owned="EXISTS (SELECT 1 FROM core_file p WHERE p.module = i.source_module)"
    )


class ProjectReachedNames(Relation):
    """`reached_names()` for every importer, a module being the project's as
    `project_imported_names()` says."""

    query = _REACHED.format(owned="EXISTS (SELECT 1 FROM core_file p WHERE p.module = i.target)")


class ReachedNames(Relation):
    """Each `binding.NAME` read of a project module's name through the module itself (`import
    pkg.limits`, `from . import limits`), with the binding, the statement that binds it under
    `statement_` columns, the importing module and its source (`content`). The module is the
    importer's own top package or reached by a relative import."""

    query = _REACHED.format(owned="i.is_project_owned AND i.target IS NOT NULL")


class NameReaders(Relation):
    """The one row holding every read of `reader_names`, top-level names of the module
    `reader_module`, in the rest of the project: the `from ... import` statements taking them
    (`importers`, with each importing file's `content`), the `binding.NAME` reads through a name
    bound to the module (`reached`), the `a.b.NAME` reads spelling the module in full (`dotted`),
    and whether some module imports every name of it (`starred`).

    The arguments are spliced into the body (see `MoveRow`): the body's aliases carry the `nr_`
    prefix and callers pass columns qualified by an alias of their own.
    """

    reader_module: str
    reader_names: list[str]

    query = r"""
        SELECT (SELECT list(struct_insert(nr_x, content := nr_c.content)
                            ORDER BY nr_x.path, nr_x.start_line, nr_x.start_column)
                FROM project_imported_names() nr_x JOIN source_files nr_c ON nr_c.path = nr_x.path
                WHERE nr_x.module = reader_module AND list_contains(reader_names, nr_x.name))
                 AS importers,
               (SELECT list(nr_r ORDER BY nr_r.path, nr_r.statement_start_line, nr_r.start_line,
                            nr_r.start_column)
                FROM project_reached_names() nr_r
                WHERE nr_r.module = reader_module AND list_contains(reader_names, nr_r.name))
                 AS reached,
               (SELECT list({'path': nr_f.path, 'name': substr(nr_n.name, strlen(reader_module) + 2),
                             'start_line': nr_n.start_line, 'start_column': nr_n.start_column,
                             'end_line': nr_n.end_line, 'end_column': nr_n.end_column,
                             'content': nr_s.content}
                            ORDER BY nr_f.path, nr_n.start_line, nr_n.start_column)
                FROM core_node nr_n
                JOIN core_file nr_f ON nr_f.file_id = nr_n.file_id AND nr_f.language = 'python'
                JOIN source_files nr_s ON nr_s.path = nr_f.path
                WHERE nr_n.type = 'attribute' AND starts_with(nr_n.name, reader_module || '.')
                  AND list_contains(reader_names, substr(nr_n.name, strlen(reader_module) + 2)))
                 AS dotted,
               EXISTS (SELECT 1 FROM core_import nr_w
                       WHERE coalesce(nr_w.source_module, nr_w.module) = reader_module
                         AND nr_w.name = '*') AS starred
    """


class MoveRow(Relation):
    """The `move-definitions` row moving `move_names` out of the module at `move_path` into the
    module at `move_destination`: the source (`module`, `content`), every `from ... import` and
    `module.NAME` read of the names (`importers`, `reached`, as `name_readers` states them), and
    the destination's dotted name and text, its name read off the source's path when the file does
    not exist yet. No row unless `move_path` is a project module.

    The arguments are spliced into the body, where an alias of the same name would capture them: the
    body's aliases carry the `mv_` prefix and callers pass columns qualified by an alias of their
    own (`finding.path`).
    """

    move_path: str
    move_names: list[str]
    move_destination: str

    query = r"""
        SELECT mv_f.path, mv_f.module, mv_s.content, move_names AS names,
               move_destination AS destination,
               coalesce(
                 (SELECT mv_d.module FROM core_file mv_d WHERE mv_d.path = move_destination),
                 array_to_string(list_filter(list_slice(
                   string_split(regexp_replace(move_destination, '\.pyi?$', ''), '/'),
                   len(string_split(regexp_replace(mv_f.path, '\.pyi?$', ''), '/'))
                     - len(string_split(mv_f.module, '.'))
                     - is_initializer_path(mv_f.path)::INTEGER + 1,
                   len(string_split(move_destination, '/'))), lambda part: part <> '__init__'),
                   '.')) AS destination_module,
               (SELECT mv_c.content FROM source_files mv_c WHERE mv_c.path = move_destination)
                 AS destination_content,
               mv_n.importers, mv_n.reached
        FROM core_file mv_f JOIN source_files mv_s ON mv_s.path = mv_f.path,
             name_readers(mv_f.module, move_names) mv_n
        WHERE mv_f.path = move_path
    """


class ModuleReaders(Relation):
    """The one row holding `files`: each Python file a move of the modules `reader_modules` touches.

    Those are the modules themselves and every file importing one (a `from ... import`, a plain
    `import`, a member that is a module) or reading one as `a.b.NAME` through a plain `import a`.
    Each is a record of its `path`, `module` and `content`, the row `move-modules` respells.

    The argument is spliced into the body (see `MoveRow`): the body's aliases carry the `rd_`
    prefix and callers pass columns qualified by an alias of their own.
    """

    reader_modules: list[str]

    query = r"""
        SELECT list({'path': rd_f.path, 'module': rd_f.module, 'content': rd_s.content}
                    ORDER BY rd_f.path) AS files
        FROM core_file rd_f JOIN source_files rd_s ON rd_s.path = rd_f.path
        WHERE rd_f.language = 'python' AND (
          list_contains(reader_modules, rd_f.module)
          OR EXISTS (SELECT 1 FROM core_import rd_i
                     WHERE rd_i.file_id = rd_f.file_id
                       AND (list_contains(reader_modules, rd_i.source_module)
                            OR list_contains(reader_modules, rd_i.target)
                            OR (rd_i.statement_type = 'import_statement'
                                AND list_contains(reader_modules, rd_i.module))))
          OR EXISTS (SELECT 1
                     FROM core_import rd_p, core_node rd_n, unnest(reader_modules) AS rd_m(module)
                     WHERE rd_p.file_id = rd_f.file_id AND rd_p.statement_type = 'import_statement'
                       AND NOT rd_p.aliased AND rd_p.name = split_part(rd_m.module, '.', 1)
                       AND rd_n.file_id = rd_f.file_id AND rd_n.type = 'attribute'
                       AND rd_n.name = rd_m.module))
    """
