"""The names an event, a Git trailer and a view share for one identity."""

from enum import StrEnum
from typing import Self

_PREFIX = "Mb-"


class Key(StrEnum):
    """One identity, spelled as the OpenTelemetry attribute where one exists.

    The member name is the field name of `Identity` and, capitalized with hyphens, the Git
    trailer (`previous_session` is `Mb-Previous-Session`).

    host: the mb alias of a machine.
    job: `<target>/<handle>`, the host a job runs on and the scheduler's handle.
    object: `sha256:<hex>`, immutable bytes named by their content.
    previous_session: the session whose thread this one continues.
    revision: a Git id, which jj shares, naming code.
    session: `<harness>:<id>`, one agent conversation as its harness numbers it.
    workspace: the `[workspace] name` of a manifest, which is the directory name when unset.
    """

    host = "host.name"
    job = "mb.job"
    object = "mb.object"
    previous_session = "session.previous_id"
    revision = "vcs.ref.head.revision"
    session = "session.id"
    workspace = "mb.workspace"

    @classmethod
    def parsed(cls, token: str) -> Self | None:
        """The key a trailer token spells, in any case; `None` for a token that is not mb's."""
        return next((key for key in cls if key.trailer.lower() == token.lower()), None)

    @property
    def trailer(self) -> str:
        """The Git trailer token, `Mb-Session` for `session`."""
        return _PREFIX + "-".join(part.capitalize() for part in self.name.split("_"))
