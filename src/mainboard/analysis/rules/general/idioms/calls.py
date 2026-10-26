"""Rules over call sites (mcmr's ALL-CALL family)."""

from ....catalog import Rule


class UncheckedResultCall(Rule, id="ALL-CALL0001"):
    """Calls to a callable whose result reports failure, with the result discarded."""

    checked_callables: tuple[str, ...] = ()
    checked_prefixes: tuple[str, ...] = ()

    query = """
        SELECT path, start_line, start_column, qualified_name
        FROM core_call
        WHERE top_type = 'expression_statement' AND top_children = 1
          AND (list_contains($checked_callables::VARCHAR[], qualified_name)
          OR len(list_filter($checked_prefixes::VARCHAR[], lambda p: starts_with(qualified_name, p))) > 0)
    """
    message = "`{qualified_name}` discards the result that reports failure"


class UnboundedBlockingCall(Rule, id="ALL-CALL0002"):
    """Calls that can wait forever because none of the bound keywords was passed."""

    bounded_callables: tuple[str, ...] = ()
    bound_names: tuple[str, ...] = ("timeout", "deadline", "duration")

    query = """
        SELECT path, start_line, start_column, qualified_name, array_to_string(list_transform(
                 $bound_names::VARCHAR[], lambda n: '`' || n || '`'), ', ') AS names
        FROM core_call c
        WHERE list_contains($bounded_callables::VARCHAR[], qualified_name)
          AND NOT EXISTS (SELECT 1 FROM core_call_arg k
                          WHERE k.file_id = c.file_id AND k.call_id = c.call_id AND k.is_keyword
                            AND list_contains($bound_names::VARCHAR[], k.name))
    """
    message = "`{qualified_name}` can block without any of {names}"
