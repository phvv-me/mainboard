"""JSON documents over HTTP, cached on disk for a day and served stale while offline."""

import os
import time
import urllib.error
import urllib.request
from contextlib import suppress
from email.message import Message
from hashlib import sha1
from pathlib import Path
from threading import Event

from patos import FrozenModel
from pydantic import PrivateAttr

from .exceptions import Unreachable

_CACHE = Path.home() / ".cache/mainboard/releases"
_JSON = "application/json"


class Web(FrozenModel):
    """JSON documents over HTTP, cached on disk."""

    cache: Path = _CACHE
    ttl_seconds: int = 24 * 3600
    timeout_seconds: int = 15
    _offline: Event = PrivateAttr(default_factory=Event)

    @staticmethod
    def _age(entry: Path) -> float | None:
        with suppress(FileNotFoundError):
            return time.time() - entry.stat().st_mtime
        return None

    @staticmethod
    def _headers(url: str, *, accept: str) -> dict[str, str]:
        """What a request for `url` sends: GitHub's API also takes `GITHUB_TOKEN` when set."""
        headers = {"Accept": accept, "User-Agent": "mainboard-release-evidence"}
        if (token := os.environ.get("GITHUB_TOKEN")) and url.startswith("https://api.github.com/"):
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _read(entry: Path, url: str) -> str:
        """A cached document; an empty entry is a cached 404."""
        if text := entry.read_text(encoding="utf-8"):
            return text
        raise urllib.error.HTTPError(url, 404, "not found (cached)", Message(), None)

    def _answer(self, request: urllib.request.Request) -> str:
        """The body the server answers `request` with, decoded."""
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read().decode()

    def _download(self, url: str, *, accept: str, entry: Path) -> str:
        """The document the network answers, a 404 cached as an empty `entry`.

        A network that does not answer is `Unreachable`, and so is every later download of the
        run, which asks it no more.
        """
        if self._offline.is_set():
            raise Unreachable(f"{url}: the network did not answer earlier in this run")
        request = urllib.request.Request(url, headers=self._headers(url, accept=accept))
        try:
            return self._answer(request)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                self._store(entry, "")
            raise
        except OSError as error:
            self._offline.set()
            raise Unreachable(f"{url}: {error}") from error

    def _store(self, entry: Path, text: str) -> None:
        self.cache.mkdir(parents=True, exist_ok=True)
        scratch = entry.with_suffix(f".{os.getpid()}.{id(text)}")
        scratch.write_text(text, encoding="utf-8")
        scratch.replace(entry)

    def get(self, url: str, *, accept: str = "") -> str:
        """The document at `url`: a fresh cached copy, else the network's, else an expired copy.

        A 404 is an answer too and is cached. Raises HTTPError when the server answers with a
        failure, Unreachable when it does not answer and nothing is cached.
        """
        accept = accept or _JSON
        entry = self.cache / sha1(f"{accept} {url}".encode(), usedforsecurity=False).hexdigest()
        age = self._age(entry)
        if age is not None and age < self.ttl_seconds:
            return self._read(entry, url)
        try:
            text = self._download(url, accept=accept, entry=entry)
        except Unreachable:
            if age is None:
                raise
            return self._read(entry, url)
        self._store(entry, text)
        return text
