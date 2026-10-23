"""The engine's commands as `mainboard rules`, and as the hidden `analyze` they answered to until
2026-10-23, kept for one release.

Cyclopts refuses to register an imported app under another name than its own, and the engine's
app is named `lint` for the harness that predates the verb, so the commands are re-registered
here under the names the CLI uses.
"""

from cyclopts import App

from .lint import app as engine

rules = App(name="rules", help=engine.help)
analyze = App(name="analyze", help=engine.help, show=False)

for command in engine:
    if not command.startswith("-"):
        for group in (rules, analyze):
            group.command(engine[command], name=command)
