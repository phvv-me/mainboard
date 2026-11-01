"""AIZK as an upstream, over `aizk_client.MemoryClient`.

This module imports the client at its top, so `outbox.Memory.aizk` loads it only when memory is
used and refuses by naming the extra when the package is absent. The client is async and every
call here is one short bounded run of it. A call that cannot reach the server, or that the server
answers with an internal error, is `Unreachable`: the MCP client reports every 5xx, a failed
authorization and a dropped connection as one internal error, and the Cloudflare tunnel that fronts
AIZK answers 530 when its connector is down, so none of them can be told apart and all of them
leave a keep queued. A stored session that expired is the same: `aizk auth login` mends it. Any
other error response, a tool's own error, and an answer outside the client's contract are
`Refused`: sending the same thing again would be refused again.
"""

import asyncio
from collections.abc import Coroutine
from typing import Self

import httpx
import httpx2
from aizk_client import (
    ArtifactReceipt,
    KeepRequest,
    LocalUpload,
    LoginRequiredError,
    MemoryClient,
    ProfileStore,
    ProtocolError,
    WriteResult,
)
from fastmcp.exceptions import ToolError
from mcp.shared.exceptions import MCPError
from mcp_types import INTERNAL_ERROR
from pydantic import ValidationError

from ..core.errors import MissionError, Refused, Unavailable, Unreachable
from .enums import Kind
from .keeps import Keep
from .upstream import Remote, Upstream

# How long one call may take: a keep may upload a file, a find searches and reads.
_FIND_SECONDS = 120.0
_KEEP_SECONDS = 60.0
_STATUS_SECONDS = 15.0
# What says the server cannot be reached now: the transports, the call's bound, a session to renew.
_DOWN = (TimeoutError, LoginRequiredError, httpx.TransportError, httpx2.TransportError)
# What says the server answered no: a tool's own error, an answer outside the client's contract.
_REFUSALS = (ToolError, ProtocolError, ValidationError)
_LOGIN = "run `aizk auth login --server URL`"


class Aizk(Upstream):
    """The AIZK server a stored client profile names."""

    @classmethod
    def open(cls) -> Self:
        """The client of the profile `aizk auth login` stored; Unavailable when there is none."""
        store = ProfileStore()
        try:
            profile = store.load()
        except FileNotFoundError as missing:
            raise Unavailable(f"{missing}; {_LOGIN}") from missing
        except ValidationError as corrupt:
            raise Unavailable(
                f"{store.path} is no AIZK client profile: move it aside and {_LOGIN}"
            ) from corrupt
        return cls(MemoryClient(profile))

    @property
    def server(self) -> str:
        return str(self.client.profile.server)

    @staticmethod
    def _answered(fault: MCPError) -> MissionError:
        """An MCP error response as what it means: an internal error waits, any other is a no."""
        if fault.code == INTERNAL_ERROR:
            return Unreachable(f"AIZK answered with an error response: {fault}")
        return Refused(f"AIZK refused it: {fault}")

    @staticmethod
    def _request(keep: Keep) -> KeepRequest:
        """The wire request of `keep`; a note travels as the companion text of its item."""
        scopes = [keep.scope] if keep.scope else None
        match keep.kind:
            case Kind.TEXT:
                text = f"{keep.content}\n\n{keep.note}" if keep.note else keep.content
                return KeepRequest(text=text, scopes=scopes)
            case Kind.URI:
                return KeepRequest(source_uri=keep.content, text=keep.note or None, scopes=scopes)
            case Kind.FILE:
                if keep.source is None:
                    raise MissionError(f"the file keep {keep.key} has no file to upload")
                upload = LocalUpload(path=keep.source, filename=keep.content)
                return KeepRequest(text=keep.note or None, scopes=scopes, upload=upload)

    @staticmethod
    def _run[T](call: Coroutine[None, None, T], seconds: float) -> T:
        """Run `call` for at most `seconds`, as `Unreachable` or `Refused` when it fails.

        It is `Unreachable` when the server cannot be reached, and `Refused` when the server
        answers no or outside the client's contract.
        """
        try:
            return asyncio.run(asyncio.wait_for(call, seconds))
        except _DOWN as fault:
            raise Unreachable(f"AIZK does not answer: {fault}") from fault
        except MCPError as fault:
            raise Aizk._answered(fault) from fault
        except _REFUSALS as fault:
            raise Refused(f"AIZK refused it: {fault}") from fault

    def __init__(self, client: MemoryClient) -> None:
        self.client = client

    def find(self, question: str, *, budget: int | None = None) -> str:
        return self._run(self.client.find(question, budget), _FIND_SECONDS)

    def keep(self, keep: Keep) -> str:
        match self._run(self.client.keep(self._request(keep)), _KEEP_SECONDS):
            case WriteResult(id=document) | ArtifactReceipt(artifact_id=document):
                return str(document)
            case other:
                raise Refused(f"AIZK kept {keep.label} without naming it: {type(other).__name__}")

    def status(self) -> Remote:
        answer = self._run(self.client.authentication_status(), _STATUS_SECONDS)
        if answer.status is None:
            return Remote(login=answer.authenticated)
        caller = answer.status.caller
        return Remote(
            login=True,
            caller=caller.name or caller.username or "",
            finds=answer.status.usage.lifetime.finds,
            keeps=answer.status.usage.lifetime.keeps,
            processing=answer.status.processing.state,
        )
