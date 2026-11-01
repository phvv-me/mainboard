"""The engine's commands as `mb code rules`, the rules that judge a project and the fixes that
repair it, and as `mb code refactor`, the tools that move, prove and map its code."""

from ...analysis.commands import catalog, gate, swift
from ...analysis.commands.graphs import diagram, matrix, uses
from ...analysis.commands.papers import impact
from ...analysis.commands.tools import move, preview, proof, sql, trial
from ...analysis.engine import lint
from ..constants import TREE

rules = TREE.node("code rules").app(
    lint.check, lint.fix, lint.test, catalog.explain, gate.gate, preview, sql, trial
)
refactor = TREE.node("code refactor").app(
    diagram.diagram,
    impact.impact,
    matrix.matrix,
    move,
    proof,
    swift.swift_compare,
    swift.swift_inventory,
    uses.uses,
)
