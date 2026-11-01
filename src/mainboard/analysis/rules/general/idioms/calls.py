"""Rules over call sites (mcmr's ALL-CALL family)."""

from ....catalog import Example, Relation, Rule

_CALLABLES = (
    '[tool.mcmr.rules."ALL-CALL0002"]\nbounded_callables = ["requests.get", "subprocess.run"]\n'
)
_POLICY = {"pyproject.toml": _CALLABLES + "bound_value = 30\n"}


class KeywordAppended(Relation):
    """The `captures` and `template` of a `rewrite-node` row that gives a Python call one more
    keyword: the argument list `appended_node` of the file `appended_path` keeps its arguments
    where they stood and takes `appended_keyword` (`timeout=30`) after the last. One row, also for
    a list with no arguments.

    The arguments are spliced into the body, where an alias of the same name would capture them:
    the body's aliases carry the `ka_` prefix and callers pass columns qualified by an alias of
    their own (`found.node`).
    """

    appended_path: str
    appended_node: int
    appended_keyword: str

    query = r"""
        SELECT MAP {'ARGS': coalesce(list(ka_a.node_id ORDER BY ka_a.sibling_index)
                                       FILTER (ka_a.node_id IS NOT NULL), []::BIGINT[])} AS captures,
               if(count(ka_a.node_id) = 0, '(', '($$$ARGS, ') || appended_keyword || ')' AS template
        FROM (SELECT 1) ka_one
        LEFT JOIN ast_nodes ka_a ON ka_a.path = appended_path AND ka_a.parent_id = appended_node
          AND ka_a.semantic_type NOT IN ('PARSER_DELIMITER', 'METADATA_COMMENT')
    """


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


class UnboundedBlockingCall(Rule, id="ALL-CALL0002", applicability="review"):
    """Calls that can wait forever because none of the bound keywords was passed.

    bound_value: the project's one timeout policy, passed as the first of `bound_names` by the
        fix; without it the rule reports and fixes nothing.
    """

    bounded_callables: tuple[str, ...] = ()
    bound_names: tuple[str, ...] = ("timeout", "deadline", "duration")
    bound_value: int | float | None = None

    query = """
        SELECT path, start_line, start_column, qualified_name, array_to_string(list_transform(
                 $bound_names::VARCHAR[], lambda n: '`' || n || '`'), ', ') AS names,
               CASE WHEN $bound_value IS NOT NULL AND len($bound_names::VARCHAR[]) > 0
                         AND language = 'python'
                         AND NOT EXISTS (SELECT 1 FROM core_call_arg s
                                         WHERE s.file_id = c.file_id AND s.call_id = c.call_id
                                           AND s.type IN ('list_splat', 'dictionary_splat'))
                    THEN (SELECT max(l.node_id) FROM core_node l
                          WHERE l.file_id = c.file_id AND l.parent_id = c.call_id
                            AND l.type = 'argument_list') END AS node
        FROM core_call c
        WHERE list_contains($bounded_callables::VARCHAR[], qualified_name)
          AND NOT EXISTS (SELECT 1 FROM core_call_arg k
                          WHERE k.file_id = c.file_id AND k.call_id = c.call_id AND k.is_keyword
                            AND list_contains($bound_names::VARCHAR[], k.name))
    """
    message = "`{qualified_name}` can block without any of {names}"
    # The call's argument list takes the bound as one more keyword, after the arguments it has. A
    # call spreading `*args` or `**kwargs` may bring the bound with it, and one whose only
    # argument is a bare generator has no argument list to extend; both are left alone, as is
    # every call of a language other than Python. A comment closing the arguments would swallow
    # the keyword, so the rewrite refuses it.
    fix = r"""
        SELECT f.finding_key, 'rewrite-node' AS operation, f.path, f.node, k.captures, k.template
        FROM finding f,
             keyword_appended(f.path, f.node,
               ($bound_names::VARCHAR[])[1] || '=' || CAST($bound_value AS VARCHAR)) k
        WHERE f.node IS NOT NULL
    """
    examples = (
        Example(
            bad="import subprocess\n\n\ndef build(cmd):\n    return subprocess.run(cmd)\n",
            good="import subprocess\n\n\ndef build(cmd):\n    return subprocess.run(cmd, timeout=30)\n",
            context=_POLICY,
        ),
        Example(
            bad="import requests\n\n\ndef fetch(url, headers):\n    return requests.get(\n        url,\n        headers=headers,\n    )\n",
            good="import requests\n\n\ndef fetch(url, headers):\n    return requests.get(url, headers=headers, timeout=30)\n",
            context=_POLICY,
        ),
        Example(
            bad="import requests\n\n\ndef fetch():\n    return requests.get()\n",
            good="import requests\n\n\ndef fetch():\n    return requests.get(timeout=30)\n",
            context=_POLICY,
        ),
        Example(
            bad="import requests\n\n\ndef fetch(url, options):\n    return requests.get(url, **options)\n",
            good="import requests\n\n\ndef fetch(url, options):\n    return requests.get(url, **options)\n",
            context=_POLICY,
        ),
        Example(
            bad="import requests\n\n\ndef fetch(url, headers):\n    return requests.get(\n        url,\n        headers=headers,  # the caller's\n    )\n",
            good="import requests\n\n\ndef fetch(url, headers):\n    return requests.get(\n        url,\n        headers=headers,  # the caller's\n    )\n",
            context=_POLICY,
        ),
        Example(
            bad="import requests\n\n\ndef fetch(url):\n    return requests.get(url)\n",
            good="import requests\n\n\ndef fetch(url):\n    return requests.get(url)\n",
            context={"pyproject.toml": _CALLABLES},
        ),
    )
