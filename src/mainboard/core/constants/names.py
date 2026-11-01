"""Names several packages spell the same way, below all of them so any may import."""

from pathlib import PurePosixPath
from typing import Final

from ..project import Project

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

# The only PATH a resident unit and a served child start with: what they run is spelled by
# absolute path.
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

SSH_FAMILY: Final = "ssh-family"

# A SHA-256 digest in lowercase hex, as every model that carries one validates it.
SHA256 = r"^[0-9a-f]{64}$"

# The option runs Codex's child carries, each contiguous: N1 of ROADMAP-v2 has it approve on
# request inside a workspace-write sandbox, and the policy's enforcer for it requires the same.
CODEX_FLAGS = (("-c", "approval_policy=on-request"), ("-c", "sandbox_mode=workspace-write"))
