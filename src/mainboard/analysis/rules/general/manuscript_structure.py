"""Rules over a manuscript's statements and floats (mcmr's ALL-MANU0004 to ALL-MANU0006)."""

from collections.abc import Sequence

from ...catalog import Rule

_SPACE = " \t\n\r\x0b\x0c"


class NumberedStatementLeftWithoutAnArgument(Rule, id="ALL-MANU0004"):
    """Statements that owe a proof and are followed by neither an argument nor an attribution."""

    discharge_heads: Sequence[str] = ("proof", "why", "sketch", "argument", "derivation")
    unproved_kinds: Sequence[str] = ("conjecture", "assumption", "problem", "question")

    query = f"""
        WITH unargued AS (
          SELECT *, trim(lower(coalesce(discharge_head, '')), '{_SPACE}') AS head
          FROM manuscript_statements
          WHERE owes_proof AND NOT list_contains($unproved_kinds::VARCHAR[], kind)
            AND proof_order = 0 AND attribution_keys_length = 0)
        SELECT path, line AS start_line, 0 AS start_column, kind, label, left(head, 40) AS head
        FROM unargued
        WHERE len(list_filter($discharge_heads::VARCHAR[], lambda w: starts_with(head, w))) = 0
    """
    message = "`{kind}` `{label}` is followed by `{head}` rather than by an argument"


class FloatWithoutATextReference(Rule, id="ALL-MANU0006"):
    """Labelled figures and tables the text never refers to."""

    query = """
        SELECT path, line AS start_line, 0 AS start_column, kind, label
        FROM manuscript_floats f
        WHERE length(label) > 0 AND NOT EXISTS (SELECT 1 FROM manuscript_references r
                                                WHERE r.fact_id = f.fact_id AND r.target = f.label)
    """
    message = "`{kind}` `{label}` is never referenced"


class NumberedStatementNothingRefersTo(Rule, id="ALL-MANU0005"):
    """Labelled statements no cross reference in the manuscript names. A measurement rather than
    a defect: a statement can be worth numbering for the reader's navigation alone, so a project
    states its own ceiling; kinds in `ignored_kinds` are illustrative and left alone."""

    ignored_kinds: Sequence[str] = ("example", "remark")

    query = """
        SELECT path, line AS start_line, 0 AS start_column, kind, label
        FROM manuscript_statements s
        WHERE length(label) > 0 AND NOT list_contains($ignored_kinds::VARCHAR[], kind)
          AND NOT EXISTS (SELECT 1 FROM manuscript_references r
                          WHERE r.fact_id = s.fact_id AND r.target = s.label)
    """
    message = "`{kind}` `{label}` is numbered but nothing refers to it"
