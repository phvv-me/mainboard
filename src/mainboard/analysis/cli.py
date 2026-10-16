"""The engine's commands as `mainboard analyze`.

Cyclopts refuses to register an imported app under another name than its own, and the engine's
app is named `lint` for the harness that predates the verb, so the commands are re-registered
here under the name the CLI uses.
"""

from cyclopts import App

from .lint import app as engine

analyze = App(name="analyze", help=engine.help)

for command in engine:
    if not command.startswith("-"):
        analyze.command(engine[command], name=command)
