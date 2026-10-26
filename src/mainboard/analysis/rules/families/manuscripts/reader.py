"""Reading LaTeX manuscripts in SQL: the element stream every manuscript family derives from.

TeX has no grammar, so the reader is lexical like mcmr's. One regular expression cuts a file into
atoms that tile it exactly (a comment, a math span, an environment, a command with the groups it
takes, a run of words), SQL classifies each atom, and the atoms become the elements a reader
meets in order: headings, labels, references, citations, mathematics, environments and the runs
of prose between them. The files a root includes are spliced in where they are included, and the
position of every element (section, statement, float, table, math, proof) falls out of window
functions over that one order.
"""

from ..constants.manuscripts.reader import (
    ARGUED_ONE,
    ARGUED_TWO,
    BRACKET,
    CITATIONS,
    DISCARDED_ONE,
    DISCARDED_THREE,
    DISCARDED_TWO,
    ENVIRONMENT_KINDS,
    GROUP,
    MACROS,
    REFERENCES,
    SECTIONS,
    THEOREMS,
)

_MATH = (
    *("align", "alignat", "displaymath", "eqnarray", "equation", "flalign", "gather", "math"),
    *("multline", "split"),
)
_VERBATIM = ("Verbatim", "lstlisting", "minted", "tikzpicture", "verbatim")
_FIGURES = ("figure", "subfigure", "wrapfigure")
_TABLES = ("table", "wraptable")
_CELLS = ("longtable", "tabular", "tabularx", "tabulary")
_EMPHASIS = ("emph", "term", "textbf", "textit", "textsc", "underline")


def _names(names: tuple[str, ...]) -> str:
    return "(?:" + "|".join(names) + ")"


def _raw(name: str) -> str:
    return rf"\\begin\{{{name}\*?\}}(?s:.*?)(?:\\end\{{{name}\*?\}}|\z)"


def atom_pattern() -> str:
    """The regular expression whose matches tile a LaTeX file.

    Each match is one thing the reader reacts to, and a control sequence takes the groups its
    meaning reads right behind it.
    """
    alternatives = [
        r"%[^\n]*\n?",
        r"\$\$(?s:.*?)(?:\$\$|\z)",
        r"\$[^$]*(?:\$|\z)",
        *map(_raw, _MATH + _VERBATIM),
        r"\\\[(?s:.*?)(?:\\\]|\z)",
        r"\\\((?s:.*?)(?:\\\)|\z)",
        rf"\\begin\{{{_names(ARGUED_ONE)}\*?\}}(?:{BRACKET})*{GROUP}",
        rf"\\begin\{{{_names(ARGUED_TWO)}\*?\}}(?:{BRACKET})*{GROUP}(?:{GROUP})?",
        rf"\\begin\*?\{{[^{{}}]*\}}(?:{BRACKET})*",
        r"\\end\*?\{[^{}]*\}",
        rf"\\{_names(SECTIONS)}\*?(?:{BRACKET})?{GROUP}",
        rf"\\label\*?{GROUP}",
        rf"\\captionof\*?{GROUP}(?:{BRACKET})?{GROUP}",
        rf"\\caption\*?(?:{BRACKET})?{GROUP}",
        rf"\\{_names(REFERENCES)}\*?(?:{BRACKET})?{GROUP}",
        rf"\\{_names(CITATIONS)}\*?(?:{BRACKET}){{0,2}}{GROUP}",
        rf"\\(?:input|include)\*?{GROUP}",
        rf"\\{_names(THEOREMS)}\*?(?:{BRACKET})?{GROUP}(?:{BRACKET})?(?:{GROUP})?(?:{BRACKET})?",
        rf"\\theoremstyle\*?{GROUP}",
        rf"\\newtheoremstyle\*?{GROUP}(?:{GROUP}){{0,8}}",
        rf"\\{_names(MACROS)}\*?(?:\s*{GROUP}|\s*\\[A-Za-z@]+)\s*(?:{BRACKET}\s*)*(?:{GROUP})?",
        rf"\\{_names(ENVIRONMENT_KINDS)}\*?{GROUP}(?:{BRACKET})*(?:{GROUP}){{0,2}}",
        rf"\\item\*?{BRACKET}",
        rf"\\bibitem\*?(?:{BRACKET}(?:{GROUP})?|{GROUP})",
        rf"\\{_names(DISCARDED_ONE)}\*?(?:{BRACKET})*{GROUP}",
        rf"\\{_names(DISCARDED_TWO)}\*?(?:{BRACKET})*{GROUP}(?:{GROUP})?",
        rf"\\{_names(DISCARDED_THREE)}\*?(?:{BRACKET})*{GROUP}(?:{GROUP}){{0,2}}",
        r"\\[A-Za-z@]+\*?",
        r"\\(?s:.)",
        r"\n\s*",
        r"[{}]",
        r"[^%\\${}\n]+",
    ]
    return "|".join(f"(?:{a})" for a in alternatives)


def quoted(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def strip(expression: str) -> str:
    """The expression without the whitespace around it (Rust's `trim`)."""
    return rf"regexp_replace({expression}, '^\s+|\s+$', '', 'g')"


def words(expression: str) -> str:
    """How many words the text holds, split at whitespace or `~`."""
    return rf"len(regexp_extract_all({expression}, '[^\s~]+'))"


def role(environment: str) -> str:
    """What an environment name is for, ignoring a starred spelling."""
    base = f"rtrim({environment}, '*')"
    lists = {
        "math": _MATH,
        "verbatim": _VERBATIM,
        "figure": _FIGURES,
        "table": _TABLES,
        "cells": _CELLS,
        "proof": ("proof",),
        "bibliography": ("thebibliography",),
        "algorithm": ("algorithm",),
    }
    branches = " ".join(
        f"WHEN {base} IN ({', '.join(map(quoted, names))}) THEN {quoted(label)}"
        for label, names in lists.items()
    )
    return f"CASE {branches} ELSE 'other' END"


def punctuation(body: str) -> str:
    """The rendered punctuation ending a display formula, '' when it ends in none.

    Control words get a marker byte behind them so `\\quad` is told from `\\quadratic` without
    lookahead.
    """
    named = rf"regexp_replace(regexp_replace({body}, '%[^\n]*', '', 'g'), '(\\[A-Za-z]+)', '\1' || chr(1), 'g')"
    skipped = rf"""regexp_replace(regexp_replace(regexp_replace({named},
        '\\(?:end|label|tag)\x01\{{[^{{}}]*\}}', '', 'g'),
        '\\(?:quad|qquad|notag|nonumber)\x01', '', 'g'),
        '\\(?:right|left)\x01\.', '', 'g')"""
    kept = rf"""regexp_replace(regexp_replace(regexp_replace({skipped},
        '\\[A-Za-z]+\x01', 'X', 'g'), '\\[^A-Za-z\x01]', '', 'g'), '[}}\s]', '', 'g')"""
    return rf"regexp_extract({kept}, '([.!?;,:])$', 1)"
