"""The engine's commands as `mainboard refactor`: the rules that judge a project, the fixes that
repair it, and the tools that move, prove and query its code."""

from cyclopts import App

from .commands import catalog, gate, swift
from .commands.graphs import diagram, matrix, uses
from .commands.papers import evidence, impact, prose
from .commands.tools import move, preview, proof, sql, trial
from .engine import lint

_HELP = "Refactor a project: check and fix it with the rules, move and prove its code."

# Every engine command; help lists them by name.
_COMMANDS = (
    lint.check,
    lint.compare,
    lint.fix,
    lint.test,
    catalog.catalog,
    catalog.explain,
    diagram.diagram,
    evidence.evidence,
    gate.gate,
    impact.impact,
    matrix.matrix,
    prose.prose,
    move,
    preview,
    proof,
    sql,
    trial,
    swift.swift_compare,
    swift.swift_inventory,
    uses.uses,
)

refactor = App(name="refactor", help=_HELP)
for command in _COMMANDS:
    refactor.command(command)
