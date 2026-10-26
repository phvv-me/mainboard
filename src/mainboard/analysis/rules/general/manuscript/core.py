"""Relations shared by the manuscript rules (mcmr's ALL-MANU family)."""

from ....catalog import Relation


class ManuscriptNumbers(Relation):
    """The `manuscript_numbers()` relation."""

    query = r"""
    SELECT *, length(literal) - length(regexp_replace(literal, '[0-9]', '', 'g')) AS digits
    FROM manuscript_evidence_numbers
"""
