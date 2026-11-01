"""What the policy decides about, and the native enforcers it checks rather than stands in for.

A provider's cap bounds a rental, a harness's own flags and settings bound an agent child, and the
agents' permission rules keep them from the owner's verbs. mb builds none of these again: a check
asks whether each is in place, and the owner's grants are the one thing mb keeps itself.
"""

from uuid import UUID

from patos import FrozenModel

from ..core.constants.names import CODEX_FLAGS

# The namespace an ask's id is minted in from the act it asks for; its event is spelled in
# `state.constants`, since the telemetry lake's `asks` view reads it too.
ASK_NAMESPACE = UUID("6f1d7c2e-3b0a-5e64-9a8d-2c4f71b0e953")

# The Claude child's own settings file, which its command names by its absolute path. It sits in
# the owner's configuration (`~/.config/mainboard/serve/<stamp>/`), outside every workspace a
# sandboxed agent may write, and is read as the owner's own 0600 file. Its writer, the
# owner-scoped `serve` after its refusal, is built with the child (N1, set aside).
CLAUDE_SETTINGS = "claude.json"

# What no allow rule of the Claude child's settings may match, besides every command at once:
# another agent, a pty that passes the owner verbs' refusal, a marker unset, or a shell of its own.
UNBOUNDED = (
    "codex exec",
    "opencode run",
    "script -q /dev/null",
    "env -u CLAUDECODE",
    "sh -c",
    "bash -c",
)

# The Codex user configuration whose values let every Codex session, not only serve's child, run
# the owner's verbs and write the grants file: no sandbox, or never asking.
CODEX_UNSANDBOXED = {"sandbox_mode": "danger-full-access", "approval_policy": "never"}

# The `serve` verbs only the owner types, after `mb serve` or `mainboard serve`: the supervisor's
# install and stop, its children run by hand, the agents' telemetry config, and leave.
_OWNER_VERBS = ("", " --*", " status*", " stop*", " agents*", " process*", " answer*", " grant*")

# The file rules that keep every agent, serve's Claude child among them, from the owner's grants
# and from the child's settings.
_KEPT = tuple(
    f"{tool}(~/.config/mainboard/{kept}/**)"
    for kept in ("grants", "serve")
    for tool in ("Read", "Edit", "Write")
)

# N2 of ROADMAP-v2, the one list the doctor counts and the owner installs in `.agents`: the agents'
# own permission rules deny the owner's verbs in every spelling a harness hands its shell, the
# grants and child settings folders, and the wrappers that pass each verb's agent-audience
# refusal (a pty from `script`, a marker turned off). That refusal is a guard; this list and the
# harness sandbox (N1(a)) are the boundary.
DENIED = (
    *(f"Bash({tool} serve{verb})" for tool in ("mb", "mainboard") for verb in _OWNER_VERBS),
    "Bash(*serve answer*)",
    "Bash(*serve grant*)",
    "Bash(*serve/keeper.py*)",
    "Bash(*launchctl bootstrap*mainboard.serve.*)",
    "Bash(*systemctl --user*mainboard-serve-*)",
    "Bash(*MB_AGENT=0*)",
    "Bash(*MAINBOARD_AGENT=0*)",
    "Bash(script *)",
    "Bash(*.config/mainboard/grants*)",
    "Bash(*.config/mainboard/serve*)",
    *_KEPT,
)


class Enforcer(FrozenModel):
    """A harness's own control over one `serve` child, declared so a launch and doctor check it.

    child: the serve child it bounds, a name in the serve registry.
    flags: the argument runs the child's command must carry, each run contiguous.
    config: the options that set one `key=value` of the harness's configuration, as `flags` sets
        its keys; none may set one of those keys again to another value.
    refused: the options the child's command never carries, whatever their value, since each
        loosens what `flags` or the settings bound.
    settings: the settings file's name in the child's folder of the owner's configuration, which
        the command names with `--settings`; empty when the flags are the whole enforcer.
    denies: the permission rules that file's `permissions.deny` must hold.
    """

    child: str
    flags: tuple[tuple[str, ...], ...] = ()
    config: tuple[str, ...] = ()
    refused: tuple[str, ...] = ()
    settings: str = ""
    denies: tuple[str, ...] = ()


# N1 of ROADMAP-v2: Codex approves on request inside a workspace-write sandbox, and the Claude
# child runs under settings of its own, which no `.agents` allow reaches. Both children are off.
ENFORCERS = (
    Enforcer(
        child="codex",
        flags=CODEX_FLAGS,
        config=("-c", "--config"),
        refused=(
            "--dangerously-bypass-approvals-and-sandbox",
            "--yolo",
            "-s",
            "--sandbox",
            "-a",
            "--ask-for-approval",
        ),
    ),
    Enforcer(
        child="claude",
        refused=(
            "--dangerously-skip-permissions",
            "--allow-dangerously-skip-permissions",
            "--permission-mode",
            "--allowedTools",
            "--allowed-tools",
        ),
        settings=CLAUDE_SETTINGS,
        denies=(
            "Bash(codex exec*)",
            "Bash(opencode run*)",
            "Read(./.env)",
            "Read(./.mb/serve/**)",
            "Read(~/.ssh/**)",
            *_KEPT,
        ),
    ),
)
