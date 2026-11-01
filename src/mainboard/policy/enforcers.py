"""Whether each native enforcer the policy relies on is in place, for a launch and for `doctor`.

Nothing here enforces: a provider's cap, a harness's flags and settings, the supervisor's
environment allowlist, the agents' deny rules, the Codex user sandbox and the owner's grants file
each do, and each is read where it lives. An agent child that is off needs nothing, which is
today's state, since N1 was set aside: the children stay off and their enforcers stay unbuilt.
"""

import os
import tomllib
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from patos import FrozenModel, FrozenOpenModel
from pydantic import ConfigDict, Field, ValidationError, field_validator

from ..core.environ import Dotenv
from ..core.errors import ExposedFile, MissionError
from ..core.section import Section
from ..core.services.base import Managed
from ..dispatch.backends import route
from ..enums import Bound, Verdict
from .constants import CODEX_UNSANDBOXED, DENIED, ENFORCERS, UNBOUNDED, Enforcer
from .grants import GrantsFile, OwnerFile, said

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence, Set
    from datetime import datetime

    from ..manifest.schema.compute.host import HostProfile
    from ..serve.constants import Service

# The verdicts in the order a row takes the worst of them.
_ORDER = (Verdict.PASS, Verdict.WARN, Verdict.FAIL)
_OFF = "not configured (children off)"
_AGENTS = Path(".agents") / "settings.json"


class Permissions(FrozenOpenModel):
    """The `permissions` of a Claude settings file, as far as the policy reads them."""

    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()


class AgentSettings(FrozenOpenModel):
    """A Claude settings file the agents run under, as far as the policy reads it."""

    permissions: Permissions = Permissions()


class ChildPermissions(Permissions):
    """The Claude child's permissions: denials of its own, no allow past them, bypass refused.

    bypass: read only as Claude spells it, since Claude reads no other spelling.
    """

    model_config = ConfigDict(validate_by_name=False)

    bypass: Literal["disable"] = Field(alias="disableBypassPermissionsMode")
    deny: tuple[str, ...]

    @field_validator("allow")
    @classmethod
    def _bounded(cls, allow: tuple[str, ...]) -> tuple[str, ...]:
        """Refuse an allow rule letting the child run any shell command, an agent or a pty."""
        if loose := [rule for rule in allow if ChildPermissions._unbounded(rule)]:
            raise ValueError(f"it allows {', '.join(loose)}, which runs past every denial")
        return allow

    @staticmethod
    def _unbounded(rule: str) -> bool:
        """Whether the allow `rule` lets a shell run anything, or one of `UNBOUNDED`'s commands."""
        if rule == "Bash":
            return True
        if not (rule.startswith("Bash(") and rule.endswith(")")):
            return False
        pattern = rule.removeprefix("Bash(").removesuffix(")")
        pattern = pattern.removesuffix(":*") + "*" if pattern.endswith(":*") else pattern
        return not pattern.strip("*") or any(fnmatchcase(run, pattern) for run in UNBOUNDED)


class ChildSettings(FrozenOpenModel):
    """The Claude child's own settings file."""

    permissions: ChildPermissions


class State(FrozenModel):
    """One enforcer as the doctor row lists it.

    said: its state in a few words.
    fix: the command that repairs it, empty when nothing needs repairing.
    """

    name: str
    verdict: Verdict
    said: str
    fix: str = ""


class Enforcement:
    """The native enforcers of one workspace, read where each lives.

    root: the workspace.
    services: the serve children declared, running or off.
    """

    @staticmethod
    def _carries(command: Sequence[str], *, run: Sequence[str]) -> bool:
        """Whether `run` appears in `command` as consecutive arguments."""
        width = len(run)
        return any(
            tuple(command[i : i + width]) == tuple(run) for i in range(len(command) - width + 1)
        )

    @staticmethod
    def _loosened(enforcer: Enforcer, command: Sequence[str]) -> list[str]:
        """The options of `command` that lift `enforcer`: a refused one, or a key set again.

        An option counts however it is spelled: `-c v`, `-cv`, `--config v` or `--config=v`.
        """
        pairs = (run[1].partition("=") for run in enforcer.flags if run[0] in enforcer.config)
        enforced = {key: value for key, _, value in pairs}
        lifting: list[str] = []
        for argument, following in zip(command, (*command[1:], ""), strict=True):
            option, inline = Enforcement._spelled(argument)
            value = inline or following
            if option in enforcer.refused:
                lifting.append(f"the bound `{argument}` lifts")
            elif option in enforcer.config and Enforcement._resets(value, enforced):
                lifting.append(f"the bound `{option} {value}` lifts")
        return lifting

    @staticmethod
    def _resets(setting: str, enforced: Mapping[str, str]) -> bool:
        """Whether `setting`, a `key=value`, gives a key of `enforced` another value."""
        key, _, value = (part.strip() for part in setting.partition("="))
        return key in enforced and value.strip("\"'") != enforced[key]

    @staticmethod
    def _spelled(argument: str) -> tuple[str, str]:
        """The option `argument` names and the value glued to it (`--config=v`, `-cv`)."""
        if argument.startswith("--"):
            option, _, glued = argument.partition("=")
            return option, glued
        return argument[:2], argument[2:]

    def __init__(self, root: Path, services: Sequence[Service]) -> None:
        self.root = root
        self.services = services

    def _agent(self, enforcer: Enforcer) -> State:
        """An agent child's enforcer: unneeded while the child is off, else in place or not."""
        service = next((s for s in self.services if s.name == enforcer.child), None)
        if service is None:
            said = "declared for no child of serve"
            return State(name=enforcer.child, verdict=Verdict.FAIL, said=said)
        if service.off:
            return State(name=enforcer.child, verdict=Verdict.PASS, said=_OFF)
        if missing := self.missing(enforcer, service.command):
            return State(
                name=enforcer.child, verdict=Verdict.FAIL, said=f"lacks {'; '.join(missing)}"
            )
        return State(name=enforcer.child, verdict=Verdict.PASS, said="in place")

    def _settings(self, enforcer: Enforcer) -> list[str]:
        """What the child's settings file lacks of `enforcer`'s denials, read as the owner's own.

        A file that is a link, another user's, open to others or of another shape lacks it all.
        """
        settings_file = self.settings_file(enforcer)
        try:
            text = settings_file.text()
        except MissionError as fault:
            return [str(fault)]
        if text is None:
            return [f"its settings file {settings_file.path}"]
        try:
            denied = ChildSettings.model_validate_json(text).permissions.deny
        except ValidationError as fault:
            error = fault.errors()[0]
            where = ".".join(str(part) for part in error["loc"])
            return [f"a settings file it may run under ({where}: {said(fault)})"]
        return [rule for rule in enforcer.denies if rule not in denied]

    def allowlist(self, allowed: Set[str]) -> State:
        """The supervisor's environment: its allowlist alone, and no child adding a `.env` name.

        allowed: the names the supervisor's whole environment holds, which every child inherits.

        Only the `.env`'s names are read, never a value.
        """
        dotenv = set(Dotenv.at(self.root).values)
        running = [service for service in self.services if not service.off]
        added = {name for service in running for name, _ in service.environment}
        if leaked := sorted(added & dotenv):
            said = f"a child declares {', '.join(leaked)}, which the .env defines"
            return State(name="allowlist", verdict=Verdict.FAIL, said=said)
        said = (
            f"{len(allowed)} names, and {len(running)} running children add {len(added)}, "
            f"none of the {len(dotenv)} the .env defines"
        )
        return State(name="allowlist", verdict=Verdict.PASS, said=said)

    def caps(self, profiles: Mapping[str, HostProfile]) -> State:
        """The cap on each provider a declared host rents from, a card one needing its limit.

        A card's limit is the owner's declaration in the grants file. A provider with no cap, or
        a grants file refused, is named with the refusal a rent through it would raise.
        """
        said: list[str] = []
        refused = False
        for alias, profile in sorted(profiles.items()):
            backend = route(profile.kind)
            if isinstance(backend, str):
                continue
            try:
                cap = backend.cap(card_usd=GrantsFile(self.root).read().card(backend.name))
            except MissionError as fault:
                refused = True
                said.append(f"{alias} refused ({fault})")
                continue
            limit = f" ${cap.limit_usd:g}" if cap.bound is Bound.CARD else ""
            said.append(f"{alias} {cap.bound}{limit}")
        if not said:
            return State(name="caps", verdict=Verdict.PASS, said="no host rents from a provider")
        verdict = Verdict.WARN if refused else Verdict.PASS
        return State(name="caps", verdict=verdict, said=", ".join(said))

    def codex_config(self) -> State:
        """The owner's Codex user configuration, which bounds every Codex session (N1(a)).

        Unsandboxed or never asking, any Codex session writes the grants file and runs the owner's
        verbs, since their refusal is a guard a same-user process steps around. Only the two keys
        of `CODEX_UNSANDBOXED` are read.
        """
        path = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"
        try:
            config = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            unset = f"no {path}, so Codex's own defaults hold"
            return State(name="codex user", verdict=Verdict.PASS, said=unset)
        except tomllib.TOMLDecodeError as fault:
            return State(name="codex user", verdict=Verdict.FAIL, said=f"{path}: {fault}")
        loose = [
            f"{key} = {value!r}"
            for key, value in CODEX_UNSANDBOXED.items()
            if config.get(key) == value
        ]
        if not loose:
            return State(name="codex user", verdict=Verdict.PASS, said="sandboxed and asking")
        danger = (
            f"{' and '.join(loose)} in {path}: any Codex session can write the grants file and "
            "run the owner's verbs"
        )
        fix = f"set sandbox_mode = 'workspace-write' and approval_policy = 'on-request' in {path}"
        return State(name="codex user", verdict=Verdict.FAIL, said=danger, fix=fix)

    def denied(self) -> State:
        """How many of N2's deny rules the agents' settings hold, which only the owner installs."""
        try:
            text = (self.root / _AGENTS).read_text(encoding="utf-8")
        except FileNotFoundError:
            return State(name="deny", verdict=Verdict.WARN, said=f"no {_AGENTS} here")
        try:
            rules = AgentSettings.model_validate_json(text).permissions.deny
        except ValidationError as fault:
            return State(name="deny", verdict=Verdict.WARN, said=f"{_AGENTS}: {said(fault)}")
        missing = [rule for rule in DENIED if rule not in rules]
        held = f"{len(DENIED) - len(missing)} of N2's {len(DENIED)} rules in {_AGENTS}"
        if missing:
            return State(name="deny", verdict=Verdict.WARN, said=f"{held}, the owner installs N2")
        return State(name="deny", verdict=Verdict.PASS, said=held)

    def grants(self, *, at: datetime) -> State:
        """The owner's grants file: absent, or the owner's own 0600 file, and what it holds."""
        grants_file = GrantsFile(self.root)
        try:
            grants = grants_file.read()
        except MissionError as fault:
            fix = (
                f"chmod 600 {grants_file.path}"
                if isinstance(fault, ExposedFile)
                else f"mv {grants_file.path} {grants_file.path}.aside"
            )
            return State(name="grants", verdict=Verdict.FAIL, said=str(fault), fix=fix)
        live = grants.live(at=at)
        listed = [f"{g.act} {g.scope} until {g.until.astimezone():%Y-%m-%d %H:%M}" for g in live]
        said = f"{len(live)} live" + (f" ({', '.join(listed)})" if listed else "")
        return State(name="grants", verdict=Verdict.PASS, said=f"{said} of {len(grants.grants)}")

    def missing(self, enforcer: Enforcer, command: Sequence[str]) -> list[str]:
        """What `command`, a child's, lacks of `enforcer`: a flag run, a settings rule, or a bound.

        A bound is lacking when an option of the command itself lifts it.
        """
        runs = list(enforcer.flags)
        if enforcer.settings:
            runs.append(("--settings", str(self.settings_file(enforcer).path)))
        lacking = [" ".join(run) for run in runs if not Enforcement._carries(command, run=run)]
        settings = self._settings(enforcer) if enforcer.settings else []
        return lacking + settings + Enforcement._loosened(enforcer, command)

    def section(
        self, profiles: Mapping[str, HostProfile], *, allowed: Set[str], at: datetime
    ) -> Section:
        """The doctor's `policy` row: every enforcer's state, the worst one its verdict.

        profiles: the declared hosts, whose providers' caps it lists.
        allowed: the names of the supervisor's environment, as `allowlist` takes them.
        """
        states = [
            *(self._agent(enforcer) for enforcer in ENFORCERS),
            self.codex_config(),
            self.allowlist(allowed),
            self.caps(profiles),
            self.denied(),
            self.grants(at=at),
        ]
        worst = max(states, key=lambda state: _ORDER.index(state.verdict))
        detail = "; ".join(f"{state.name}: {state.said}" for state in states)
        fix = next((state.fix for state in states if state.fix), "")
        return Section(section="policy", verdict=worst.verdict, detail=detail, fix=fix)

    def settings_file(self, enforcer: Enforcer) -> OwnerFile:
        """The child's settings file, in this workspace's folder of the owner's configuration."""
        return OwnerFile("serve", Managed(self.root).stamp, enforcer.settings)
