"""Rules over function signatures (mcmr's PY-FUNC family)."""

from ...catalog import Example, Rule

# One piece of a parameter list at its top level: a character outside brackets, strings and
# comments, or a bracketed group holding one more bracketed group at most.
PARAMETER_LIST_ATOM = (
    r"""(?:[^()\[\]{}"'#]|[(\[{](?:[^()\[\]{}"'#]|[(\[{][^()\[\]{}"'#]*[)\]}])*[)\]}])"""
)


class UnjustifiedPositionalOnlyParameterCount(Rule, id="PY-FUNC0001", applicability="safe"):
    """Positional-only parameters without an observable structural reason."""

    allowed_names: tuple[str, ...] = ()

    query = """
            SELECT f.*, n.value FROM function_functions f
            JOIN (SELECT function_id, count(*) AS value FROM function_parameters
                  WHERE is_positional_only AND NOT is_receiver
                    AND NOT is_required_by_external_contract
                    AND NOT list_contains($allowed_names::VARCHAR[], name)
                  GROUP BY function_id) n ON n.function_id = f.entity_id
            WHERE f.is_protocol_member IS NOT TRUE
        """
    message = "unjustified positional only parameter count is {value} for `{fact_id}`"
    # The `/` goes with the comma before it, and callers gain the keyword form. Only when every
    # positional-only parameter lacks a justification (removing the marker would drop the others'
    # too) and no `**keywords` could collide with a name. The marker is found by reading the
    # parameter list as text, so a list holding a string, a comment or deeper brackets is left
    # alone. Offsets are characters into the definition's text, turned into line and column.
    fix = (
        r"""
        WITH unjustified AS (
          SELECT f.*, regexp_extract(f.definition_text,
                   '(?s)^.*?\bdef[ \t]+' || f.name || '[ \t]*\(') AS opening
          FROM finding f
          WHERE NOT EXISTS (
            SELECT 1 FROM function_parameters p
            WHERE p.function_id = f.entity_id AND p.is_positional_only
              AND NOT p.is_receiver AND p.is_required_by_external_contract)
        ), located AS (
          SELECT *, regexp_extract(substr(definition_text, length(opening) + 1),
            '^({atom}*?)(,[ \t\r\n]*/)({atom}*?)\)', ['skipped', 'marker', 'tail']) AS found
          FROM unjustified
        ), spans AS (
          SELECT *, length(opening) + length(found.skipped) AS marker_at
          FROM located WHERE found.marker <> '' AND NOT contains(found.tail, '**')
        ), cuts AS (
          SELECT *, string_split(substr(definition_text, 1, marker_at), chr(10)) AS up_to,
                 string_split(substr(definition_text, 1, marker_at + length(found.marker)),
                              chr(10)) AS through
          FROM spans
        )
        SELECT finding_key, definition_path AS path,
               definition_start_line + len(up_to) - 1 AS start_line,
               if(len(up_to) = 1, definition_start_column, 0) + strlen(up_to[-1]) AS start_column,
               definition_start_line + len(through) - 1 AS end_line,
               if(len(through) = 1, definition_start_column, 0) + strlen(through[-1])
                 AS end_column,
               '' AS text
        FROM cuts
    """
    ).replace("{atom}", PARAMETER_LIST_ATOM.replace("'", "''"))
    examples = (
        Example(
            bad="def load(path=None, /):\n    return path\n",
            good="def load(path=None):\n    return path\n",
        ),
    )
