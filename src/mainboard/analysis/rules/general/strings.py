"""Rules over string expressions (mcmr's ALL-STRI family)."""

from ...catalog import Example, Rule


class FragmentedMultilineLiteral(Rule, id="ALL-STRI0001", applicability="safe"):
    """Multiline values written as several adjacent string literal fragments."""

    minimum_fragments: int = 2

    query = """
        SELECT path, start_line, start_column, end_line, end_column, value AS runtime_value,
               fragments AS literal_fragment_count, language
        FROM core_text
        WHERE kind = 'string' AND parses AND plain AND repeat_id IS NULL
          AND parent_type IS DISTINCT FROM 'expression_statement'
          AND fragments >= $minimum_fragments AND contains(value, chr(10))
    """
    message = "Multiline value is split across {literal_fragment_count} adjacent literal fragments"
    # One triple-quoted literal of the same value, in the quote the value does not use itself.
    # Python only, and only for a value that needs no escape: a backslash, carriage return or
    # control character would be read differently (or not at all) inside the triple quotes.
    fix = r"""
        SELECT finding_key, path, start_line, start_column, end_line, end_column,
               quote || runtime_value || quote AS text FROM (
          SELECT *, (SELECT repeat(mark, 3) FROM unnest([chr(34), chr(39)]) AS m(mark)
                     WHERE NOT contains(runtime_value, repeat(mark, 3))
                       AND NOT ends_with(runtime_value, mark) LIMIT 1) AS quote
          FROM finding
          WHERE language = 'python' AND NOT regexp_matches(runtime_value, '[\\\x00-\x08\x0b-\x1f]')
        ) WHERE quote IS NOT NULL
    """
    examples = (
        Example(
            bad='message = (\n    "first line\\n"\n    "second line"\n)\n',
            good='message = (\n    """first line\nsecond line"""\n)\n',
        ),
    )


class DecorativeRepeatedSeparatorCount(Rule, id="ALL-STRI0002"):
    """Separator lines built by repeating one punctuation character."""

    minimum_repetitions: int = 3

    query = r"""
        SELECT t.path, b.start_line, b.start_column, t.value AS literal,
               t.repeat_count AS repetition_count
        FROM core_text t JOIN core_node b ON b.file_id = t.file_id AND b.node_id = t.repeat_id
        WHERE t.parses AND regexp_matches(t.value, '^[-_*=~.#]+$')
          AND t.repeat_count >= $minimum_repetitions
    """
    message = "Decorative separator repeats `{literal}` {repetition_count} times"
