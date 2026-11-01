"""The relations under mcmr's ALL-OVER rules: each method a subclass overrides, its parameters
stacked and compared slot by slot."""

from .....catalog import Relation

_PLACEHOLDER = r"^(?:_+$|_[a-zA-Z0-9_]*[a-zA-Z0-9]$|dummy|ignored_|unused_)"


def tail(dotted: str) -> str:
    """The last dotted segment of `dotted`, SQL: `a.b.c` reads `c`."""
    return rf"regexp_replace({dotted}, '^.*\.', '')"


_FLAGS = rf"""flags AS (
      SELECT fact_id, parent_id, split_part(relation, '.', 1) AS side,
             bool_or(tail = 'classmethod') AS is_classmethod,
             bool_or(ends_with(string_value, '.setter')) AS is_setter,
             bool_or(tail IN ('property', 'cached_property')
                     OR regexp_matches(string_value, '\.(setter|getter|deleter)$'))
               AS is_read_as_data
      FROM (SELECT *, {tail("string_value")} AS tail FROM override_values
            WHERE relation IN ('declared.decorators', 'inherited.decorators'))
      GROUP BY ALL)"""


class OverridesPairs(Relation):
    """The `overrides_pairs()` relation."""

    query = f"""
    WITH {_FLAGS}
    SELECT fact_id, name, i.record_id AS inherited_record_id, d.record_id AS declared_record_id,
           i.parameters_present AS inherited_callable, d.parameters_present AS declared_callable,
           i.asynchronous AS inherited_asynchronous, d.asynchronous AS declared_asynchronous,
           coalesce(fi.is_classmethod, false) AS inherited_is_classmethod,
           coalesce(fd.is_classmethod, false) AS declared_is_classmethod,
           coalesce(fd.is_setter, false) AS declared_is_setter,
           coalesce(fi.is_read_as_data, false) AS inherited_is_read_as_data,
           coalesce(fd.is_read_as_data, false) AS declared_is_read_as_data
    FROM override_inherited i JOIN override_declared d USING (fact_id, name)
    LEFT JOIN flags fi
      ON fi.fact_id = i.fact_id AND fi.parent_id = i.record_id AND fi.side = 'inherited'
    LEFT JOIN flags fd
      ON fd.fact_id = d.fact_id AND fd.parent_id = d.record_id AND fd.side = 'declared'
"""


class OverridesCallablePairs(Relation):
    """The `overrides_callable_pairs()` relation: each overridden method both classes declare with
    parameters, special methods, mangled names and property setters aside."""

    query = """
    SELECT fact_id, name AS method, inherited_record_id AS inherited,
           declared_record_id AS declared, inherited_is_classmethod, declared_is_classmethod
    FROM overrides_pairs()
    WHERE inherited_callable AND declared_callable AND NOT starts_with(name, '__')
      AND NOT declared_is_setter
"""


class OverridesNumbered(Relation):
    """The `overrides_numbered()` relation: both sides' parameters of each pair, stacked and
    numbered within their kind."""

    query = """
    WITH stacked AS (
      SELECT p.fact_id, p.method, p.inherited, p.declared, 'inherited' AS side,
             p.inherited_is_classmethod AS is_classmethod, q.ordinal, q.name, q.kind, q.has_default
      FROM overrides_callable_pairs() p JOIN override_inherited_parameters q
        ON q.fact_id = p.fact_id AND q.parent_id = p.inherited
      UNION ALL
      SELECT p.fact_id, p.method, p.inherited, p.declared, 'declared', p.declared_is_classmethod,
             q.ordinal, q.name, q.kind, q.has_default
      FROM overrides_callable_pairs() p JOIN override_declared_parameters q
        ON q.fact_id = p.fact_id AND q.parent_id = p.declared
    )
    SELECT *, row_number() OVER (PARTITION BY fact_id, inherited, declared, side, kind
                                 ORDER BY ordinal) AS position
    FROM stacked
"""


class OverridesSlots(Relation):
    """The `overrides_slots()` relation: per pair and parameter slot (a position within a kind, or
    a keyword name), what the base held there and what the subclass answers, whether the slot is
    unmatched and whether the subclass binds it under another name."""

    query = rf"""
    WITH slots AS (
      SELECT *, side = 'declared' AS is_declared,
             bool_or(side = 'declared') OVER (
               PARTITION BY fact_id, inherited, declared, kind, name) AS answered_by_name,
             bool_or(side = 'declared' AND kind = 'var_positional') OVER (
               PARTITION BY fact_id, inherited, declared) AS has_varargs,
             bool_or(side = 'declared' AND kind = 'var_keyword') OVER (
               PARTITION BY fact_id, inherited, declared) AS has_kwargs
      FROM overrides_numbered()
      WHERE NOT (kind = 'positional_or_keyword' AND is_classmethod AND position = 1)
    ), held AS (
      SELECT *,
             CASE WHEN kind = 'keyword_only' THEN name ELSE CAST(position AS VARCHAR) END AS slot,
             NOT is_declared AND NOT coalesce(
               CASE WHEN kind IN ('positional_only', 'positional_or_keyword') THEN has_varargs
                    WHEN kind = 'keyword_only' THEN has_kwargs ELSE false END
               AND NOT answered_by_name, false) AS held
      FROM slots
    ), compared AS (
      SELECT fact_id, method, inherited, declared, kind, slot, bool_or(held) AS held,
             bool_or(is_declared) AS answered,
             bool_or(is_declared AND has_default) AS answer_default,
             bool_or(NOT is_declared AND has_default) AS held_default,
             max(CASE WHEN held THEN name END) AS held_name,
             max(CASE WHEN is_declared THEN name END) AS answered_name
      FROM held GROUP BY ALL
    )
    SELECT *, (held AND NOT answered) OR (NOT held AND answered AND NOT answer_default)
                AS unmatched,
           kind = 'positional_or_keyword' AND held AND answered AND held_name <> answered_name
             AND NOT regexp_matches(held_name, '{_PLACEHOLDER}')
             AND NOT regexp_matches(answered_name, '{_PLACEHOLDER}') AS renamed
    FROM compared
"""


class OverridesSignatures(Relation):
    """The `overrides_signatures()` relation."""

    query = """
    WITH optional AS (
      SELECT fact_id, inherited, declared,
             count(*) FILTER (WHERE side = 'inherited' AND has_default) AS inherited_optional,
             count(*) FILTER (WHERE side = 'declared' AND has_default) AS declared_optional
      FROM overrides_numbered() WHERE kind IN ('positional_only', 'positional_or_keyword')
      GROUP BY ALL
    ), measured AS (
      SELECT fact_id, inherited, declared,
        bool_or(unmatched AND kind = 'positional_or_keyword') AS positions_changed,
        bool_or(unmatched AND kind = 'positional_only') AS slots_changed,
        bool_or(unmatched AND kind = 'keyword_only') AS keywords_changed,
        bool_or(kind IN ('var_positional', 'var_keyword') AND held AND NOT answered)
          AS variadics_removed,
        count(*) FILTER (WHERE renamed) AS renamed_total,
        bool_or(kind = 'var_positional' AND answered) AS declared_varargs
      FROM overrides_slots()
      GROUP BY ALL
    )
    SELECT fact_id, method, inherited, declared, differing_arguments, renamed_parameters,
           CAST(differing_arguments = 0 AND renamed_parameters = 0 AND NOT declared_varargs
                AND declared_optional < inherited_optional AS BIGINT)
             AS required_what_the_base_defaulted
    FROM (SELECT p.fact_id, p.method, p.inherited, p.declared,
                 coalesce(CAST(m.positions_changed OR m.slots_changed OR m.keywords_changed
                               AS BIGINT) + CAST(m.variadics_removed AS BIGINT), 0)
                   AS differing_arguments,
                 CASE WHEN m.positions_changed THEN 0 ELSE coalesce(m.renamed_total, 0) END
                   AS renamed_parameters,
                 coalesce(m.declared_varargs, false) AS declared_varargs,
                 coalesce(o.inherited_optional, 0) AS inherited_optional,
                 coalesce(o.declared_optional, 0) AS declared_optional
          FROM overrides_callable_pairs() p
          LEFT JOIN measured m USING (fact_id, inherited, declared)
          LEFT JOIN optional o USING (fact_id, inherited, declared))
"""


class OverridesRenames(Relation):
    """The `overrides_renames()` relation: each parameter an override binds under another name
    than the base used, in a pair whose positions otherwise line up."""

    query = """
    SELECT fact_id, method, held_name AS base_name, answered_name AS own_name
    FROM (SELECT *, bool_or(unmatched AND kind = 'positional_or_keyword')
                      OVER (PARTITION BY fact_id, inherited, declared) AS positions_changed
          FROM overrides_slots())
    WHERE renamed AND NOT positions_changed
"""
