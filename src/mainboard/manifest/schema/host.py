from typing import Self

from ...core.base import Declared
from .observe import Observe
from .queue import Defaults, QueuePolicy


class Sync(Declared):
    """What ships to a remote host and what the mirror may never delete."""

    include: list[str] = []
    exclude: list[str] = []
    protect: list[str] = []

    def merged(self, over: Self) -> Self:
        """This sync scope layered over `over`, lists unioned in order.

        Layering unions rather than replaces, so a host adding one exclude
        never silently drops the workspace-wide protect rules, the footgun the
        previous generation shipped.

        over: the lower-precedence sync scope being overlaid.
        """
        return type(self)(
            include=_union(over.include, extra=self.include),
            exclude=_union(over.exclude, extra=self.exclude),
            protect=_union(over.protect, extra=self.protect),
        )


class HostProfile(Declared):
    """One remote (or local) machine's execution profile.

    Everything the previous generation scattered across `lote.toml` hints,
    global chefe `[modules]`, prose skill files, and userland constants: which
    scheduler kind, which env and container, the module stack, queue policies,
    submit defaults, sync scope, and host variables. A profile inherits the
    `[hosts.defaults]` table field-by-field before its own keys apply.
    """

    kind: str = "auto"
    root: str = ""
    account: str = ""
    login_shell: bool = True
    env: str = "default"
    container: str = ""
    modules: dict[str, str] = {}
    scratch: str = ""
    vars: dict[str, str] = {}
    sync: Sync = Sync()
    queues: dict[str, QueuePolicy] = {}
    defaults: Defaults = Defaults()
    observe: Observe = Observe()

    def inheriting(self, base: Self) -> Self:
        """This profile with `base` filling every field the profile left unset.

        base: the `[hosts.defaults]` profile being inherited from.
        """
        fields = self.model_dump(exclude_unset=True)
        fields["sync"] = self.sync.merged(base.sync)
        fields["modules"] = {**base.modules, **self.modules}
        fields["vars"] = {**base.vars, **self.vars}
        fields["queues"] = {**base.queues, **self.queues}
        merged = {**base.model_dump(exclude_unset=True), **fields}
        return type(self).model_validate(merged)

    def policy(self, queue: str) -> QueuePolicy:
        """The declared policy for `queue`, permissive when the host names none.

        queue: the scheduler queue being targeted.
        """
        return self.queues.get(queue, QueuePolicy())


def _union(base: list[str], *, extra: list[str]) -> list[str]:
    return list(dict.fromkeys([*base, *extra]))
