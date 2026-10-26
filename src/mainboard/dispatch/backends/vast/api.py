"""Vast.ai's REST API under a console key sent as `Authorization: Bearer` on every call."""

import json
from time import sleep
from typing import TYPE_CHECKING
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request

from ....core.errors import MissionError
from ....log import logger
from ..base import Credentials, forgotten, http_transport

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ..base import Transport

# Seconds between creates Vast rate-limited (HTTP 429), before the last ask whose refusal stands.
_RATE_LIMIT_WAITS = (2.0, 5.0, 10.0)
_LOG_TAIL_LINES = 2000
# `request_logs` answers before the log reaches storage, so the fetch is retried before the url is
# handed to the caller instead.
_LOG_ATTEMPTS = 20
_LOG_POLL_SECONDS = 1.0


def api_key() -> str:
    """The Vast key from `VAST_API_KEY` or `VASTAI_API_KEY`, refusing with a setup hint when unset.

    A console key authenticates the whole v0 namespace as a Bearer header, as their CLI sends it.
    gpuhunt reads `VASTAI_API_KEY` while Vast's CLI documents `VAST_API_KEY`, so both are read.
    """
    return Credentials().key(
        "VAST_API_KEY",
        "VASTAI_API_KEY",
        hint="an API key minted at https://cloud.vast.ai/manage-keys/",
    )


class VastApi:
    """The calls a rental makes, each addressing an instance by its contract id.

    transport: sends a prepared `Request`, returning its response, injectable for tests.
    sleeper: waits between polls, injected so a test drives it without real time.
    """

    def __backoff(self, wait: float) -> None:
        """Wait `wait` seconds out after a rate-limited create, saying so."""
        logger.warning("vast rate-limited the create; asking again in {}s", wait)
        self.sleeper(wait)

    def __init__(
        self, *, transport: Transport = http_transport, sleeper: Callable[[float], None] = sleep
    ) -> None:
        self.transport = transport
        self.sleeper = sleeper

    def created(self, offer: Mapping, body: dict) -> dict:
        """The create for `offer`, waited out and asked again while Vast answers 429.

        A rent that tries several taken offers in a row reaches Vast's create rate limit (HTTP
        429 on the third, 2026-09-29), which says "not yet" rather than "no", so it waits.
        """
        for wait in _RATE_LIMIT_WAITS:
            try:
                return self.request("PUT", path=f"/asks/{offer['id']}/", body=body)
            except HTTPError as refused:
                if refused.code != 429:
                    raise
                self.__backoff(wait)
        return self.request("PUT", path=f"/asks/{offer['id']}/", body=body)

    def instance(self, handle: str) -> dict:
        """`handle`'s instance row, empty once Vast has forgotten it (a null row or a 404)."""
        try:
            payload = self.request("GET", path=f"/instances/{handle}/", query={"owner": "me"})
        except HTTPError as error:
            return forgotten(error)
        return payload.get("instances") or {}

    def logs(self, handle: str) -> str:
        """The tail of `handle`'s log, which Vast uploads to storage and links."""
        payload = self.request(
            "PUT",
            path=f"/instances/request_logs/{handle}/",
            body={"tail": str(_LOG_TAIL_LINES)},
        )
        url = str(payload.get("result_url") or "")
        if not url:
            raise MissionError(
                f"vast refused logs for instance {handle}: {payload.get('msg') or payload}"
            )
        return self.uploaded(url)

    def request(
        self, method: str, *, path: str, body: dict | None = None, query: dict | None = None
    ) -> dict:
        """An authenticated call below `https://console.vast.ai/api/v0`, their CLI's default root.

        path: the endpoint below the v0 root, trailing slash included.
        body: the JSON payload, an empty object when the endpoint takes none.
        """
        tail = f"{path}?{urlencode(query)}" if query else path
        request = Request(
            f"https://console.vast.ai/api/v0{tail}",
            method=method,
            data=json.dumps(body or {}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"},
        )
        return json.loads(self.transport(request))

    def uploaded(self, url: str) -> str:
        """The log Vast uploaded at `url`, polled while the upload is in flight.

        The first fetches 404 until the log lands. The url is storage's own signed link, so it
        must be https and is fetched with no API key attached.
        """
        host = url.removeprefix("https://")
        if host == url:
            raise MissionError(f"vast answered with a non-https log url {url!r}")
        for _ in range(_LOG_ATTEMPTS):
            try:
                return self.transport(Request(f"https://{host}")).decode()
            except HTTPError:
                self.sleeper(_LOG_POLL_SECONDS)
        raise MissionError(f"vast has not uploaded that log yet; fetch {url} directly")
