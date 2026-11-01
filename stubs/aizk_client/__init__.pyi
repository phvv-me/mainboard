# The `aizk_client` distribution's surface that `mainboard.memory.aizk` calls, from the client the
# AIZK repo ships as `aizk_client`, whose models derive patos' `FrozenModel`; the optional `memory`
# extra installs the real package.

from pathlib import Path

from patos import FrozenModel
from pydantic import AnyHttpUrl

from .wire import ArtifactReceipt as ArtifactReceipt
from .wire import KeepResult, StatusReport
from .wire import UploadTicketAccepted as UploadTicketAccepted
from .wire import WriteResult as WriteResult

class LoginRequiredError(PermissionError): ...
class ProtocolError(RuntimeError): ...

class ClientProfile(FrozenModel):
    server: AnyHttpUrl
    auth: str

class ProfileStore:
    path: Path
    def __init__(self, path: Path | None = None) -> None: ...
    def load(self) -> ClientProfile: ...

class LocalUpload(FrozenModel):
    path: Path
    media_type: str | None = None
    filename: str | None = None

class KeepRequest(FrozenModel):
    text: str | None = None
    source_uri: str | None = None
    scopes: list[str] | None = None
    upload: LocalUpload | None = None

class AuthenticationStatus(FrozenModel):
    authenticated: bool
    status: StatusReport | None

class MemoryClient:
    profile: ClientProfile
    def __init__(self, profile: ClientProfile) -> None: ...
    async def authentication_status(self, days: int = 30) -> AuthenticationStatus: ...
    async def find(
        self, query: str, budget: int | None = None, scopes: list[str] | None = None
    ) -> str: ...
    async def keep(self, request: KeepRequest) -> KeepResult: ...
