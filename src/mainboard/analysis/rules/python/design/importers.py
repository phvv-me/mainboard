"""Who reads a project module's top-level names: the relations a move of definitions rewrites."""

from ....catalog import Relation


class ImportedNames(Relation):
    """Each name a project `from ... import` statement takes from a project module, with the
    importing module (`importer`), the package it imports from (`importer_package`, the module
    itself for an `__init__.py`) and the statement's own lines (`statement`)."""

    query = """
        SELECT coalesce(i.source_module, i.module) AS module,
               if(i.imported_name <> '', i.imported_name, i.name) AS name,
               i.path, i.start_line, i.start_column, i.end_line, i.end_column,
               f.module AS importer,
               if(ends_with(i.path, '__init__.py'), string_split(f.module, '.'),
                  list_slice(string_split(f.module, '.'), 1, -2)) AS importer_package,
               array_to_string(
                 list_slice(string_split(s.content, chr(10)), i.start_line, i.end_line),
                 chr(10)) AS statement
        FROM core_import i
        JOIN core_file f ON f.file_id = i.file_id
        JOIN source_files s ON s.path = i.path
        WHERE i.statement_type = 'import_from_statement' AND i.is_project_owned
    """


class ReachedNames(Relation):
    """Each `binding.NAME` read of a project module's name through the module itself (`import
    pkg.limits`, `from . import limits`), with the binding, the statement that binds it under
    `statement_` columns, the importing module and its source (`content`)."""

    query = """
        SELECT i.target AS module, substr(n.name, strlen(i.name) + 2) AS name, f.path,
               n.start_line, n.start_column, n.end_line, n.end_column,
               f.module AS importer,
               if(ends_with(f.path, '__init__.py'), string_split(f.module, '.'),
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
        WHERE i.is_project_owned AND i.target IS NOT NULL
    """
