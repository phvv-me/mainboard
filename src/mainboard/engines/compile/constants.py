"""The constants of `compile` other modules share."""

from pathlib import PurePosixPath

from ...core.project import Project

# A finished prefix's stamp naming its digest, so an interrupted build never passes as complete.
# A marker file (`Project.marker`): written under the legacy name, read under any.
STAMPED = "prefix"

ACTIVATION = "activate.sh"

# Pixi's required pair, the core of an environment's identity.
MANIFEST = "pixi.toml"

LOCK = "pixi.lock"

# Where a generated Pixi file sits below the workspace: only its depth enters the path arithmetic,
# and every state directory name the tool answers to sits at the same depth.
GENERATED_DIR = PurePosixPath(Project().out_dirs[0])
