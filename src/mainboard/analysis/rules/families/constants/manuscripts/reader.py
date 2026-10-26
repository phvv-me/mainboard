"""The constants of `manuscripts_reader` other modules share."""

SECTIONS = ("part", "chapter", "section", "subsection", "subsubsection", "paragraph")

REFERENCES = (
    "Cref",
    "Crefrange",
    "autoref",
    "cref",
    "crefrange",
    "eqref",
    "nameref",
    "pageref",
    "proofref",
    "ref",
)

CITATIONS = (
    "cite",
    "citealp",
    "citealt",
    "citeauthor",
    "citep",
    "citet",
    "citeyear",
    "citeyearpar",
)

MACROS = ("DeclareMathOperator", "newcommand", "providecommand", "renewcommand")

ENVIRONMENT_KINDS = ("newenvironment", "renewenvironment", "provideenvironment")

THEOREMS = ("newtheorem", "newmdtheoremenv")

DISCARDED_ONE = (
    "@starttoc",
    "Needspace",
    "bibliography",
    "bibliographystyle",
    "captionsetup",
    "colorbox",
    "documentclass",
    "graphicspath",
    "hspace",
    "hypersetup",
    "includegraphics",
    "pagenumbering",
    "textcolor",
    "typeout",
    "usepackage",
    "vspace",
)

DISCARDED_TWO = (
    "addtolength",
    "crefalias",
    "numberwithin",
    "pdfbookmark",
    "setcounter",
    "setlength",
)

DISCARDED_THREE = ("addcontentsline", "definecolor")

DISCARDED = DISCARDED_ONE + DISCARDED_TWO + DISCARDED_THREE

MARK = "chr(183)"

_ESCAPE = r"\\(?s:.)"

_DEPTH = 8


def _balanced(levels: int) -> str:
    """A run of text with balanced brace groups nested up to `levels` deep."""
    inside = rf"(?:[^{{}}\\]|{_ESCAPE})*"
    for _ in range(levels):
        inside = rf"(?:[^{{}}\\]|{_ESCAPE}|\{{{inside}\}})*"
    return inside


# The inside of a brace group (nested to _DEPTH levels) and of a bracket group.
INSIDE = _balanced(_DEPTH - 1)

GROUP = rf"\{{{INSIDE}\}}"

BRACKET = rf"\[(?:[^\[\]\\]|{_ESCAPE}|\[[^\[\]]*\])*\]"

BREAKS = (
    "par",
    "State",
    "Statex",
    "Require",
    "Ensure",
    "Return",
    "For",
    "EndFor",
    "If",
    "Else",
    "ElsIf",
    "EndIf",
    "While",
    "EndWhile",
    "Function",
    "EndFunction",
    "Procedure",
    "EndProcedure",
)

# Environments whose opening takes mandatory arguments (layout, never prose).
ARGUED_ONE = ("adjustbox", "array", "longtable", "minipage", "tabular")

ARGUED_TWO = ("tabularx", "tabulary")
