# The user's ssh client config, cut to the hosts a migration needs and respelled for the
# destination.

from collections.abc import Sequence
from fnmatch import fnmatch

# The macOS-only ssh setting other clients refuse unless told to ignore it.
_MACOS_ONLY = ("usekeychain",)


class SshConfig:
    """The user's ssh client config `text`, cut to the blocks this workspace's hosts need."""

    @staticmethod
    def _blocks(text: str) -> list[list[str]]:
        """The config split at each `Host` or `Match` line, leading lines their own block."""
        blocks: list[list[str]] = [[]]
        for line in text.splitlines():
            if line.strip().lower().startswith(("host ", "match ")):
                blocks.append([])
            blocks[-1].append(line)
        return [block for block in blocks if any(line.strip() for line in block)]

    @staticmethod
    def _hops(block: Sequence[str]) -> set[str]:
        """The aliases a block's jump settings go through."""
        hops: set[str] = set()
        for line in block:
            key, _, value = line.strip().partition(" ")
            if key.lower() == "proxyjump":
                hops |= {
                    hop.split("@")[-1].split(":")[0] for hop in value.split(",") if hop.strip()
                }
            elif key.lower() == "proxycommand":
                words = value.split()
                hops |= {
                    words[index + 1] for index, word in enumerate(words[:-1]) if word == "ssh"
                }
        return {hop.strip() for hop in hops if hop.strip() and hop.strip().lower() != "none"}

    @staticmethod
    def _patterns(block: Sequence[str]) -> list[str]:
        """The host patterns a `Host` block answers to, none for a `Match` or leading block."""
        head = block[0].split()
        return head[1:] if head and head[0].lower() == "host" else []

    @staticmethod
    def identities(blocks: Sequence[list[str]]) -> list[str]:
        """Every `IdentityFile` the blocks name, as written."""
        return [
            line.split(None, 1)[1].strip().strip('"')
            for block in blocks
            for line in block
            if line.strip().lower().startswith("identityfile") and len(line.split(None, 1)) > 1
        ]

    @staticmethod
    def rendered(blocks: Sequence[list[str]], *, system: str, user: str = "") -> list[str]:
        """Each block as text the destination's ssh client accepts.

        user: the login a block naming none reaches its host as, empty to leave it to ssh; the
            catch-all `Host *` is left alone, since it would rename every other host there too.
        """
        refused = _MACOS_ONLY if system != "Darwin" else ()
        rendered = []
        for block in blocks:
            lines = [line for line in block if not line.strip().lower().startswith(refused)]
            if (
                user
                and SshConfig._patterns(block) != ["*"]
                and not any(line.strip().lower().split()[:1] == ["user"] for line in block)
            ):
                lines.insert(1, f"    User {user}")
            rendered.append("\n".join(lines).strip())
        return rendered

    def __init__(self, text: str) -> None:
        self.blocks = self._blocks(text)

    def needed(self, aliases: Sequence[str]) -> list[list[str]]:
        """The blocks reaching any of `aliases`, with every jump host they go through.

        A block is kept when one of its patterns matches an alias, and the aliases its
        `ProxyJump` or `ProxyCommand ssh <alias>` go through are followed until nothing new is
        named, so a host behind a bastion arrives with its bastion.
        """
        wanted = set(aliases)
        kept: list[list[str]] = []
        grew = True
        while grew:
            grew = False
            for block in self.blocks:
                if block in kept or not any(
                    fnmatch(alias, pattern)
                    for alias in wanted
                    for pattern in SshConfig._patterns(block)
                ):
                    continue
                kept.append(block)
                wanted |= SshConfig._hops(block)
                grew = True
        return [block for block in self.blocks if block in kept]
