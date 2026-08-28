# WHERE A READING WAS TAKEN, PROBED ONCE AND STAMPED ON EVERY RECEIPT OF A RUN.
#
# Everything here is DERIVED. A fact a test has to retype is a fact a test will eventually retype
# wrong, and two harnesses that each spell their own provenance produce two receipts that cannot
# be compared on the axis that matters most, so this is the one probe and the one shape.
#
# THE CARD IS IDENTIFIED BY ITS UUID AND NOT BY ITS NAME. A name is a model number: two identical
# 4090s in one box answer `NVIDIA GeForce RTX 4090` to the same question, so a lane satisfied on
# the first would read complete on the second and publish one card's rows twice. The UUID is the
# device's identity and is what `card` carries, which makes it the coverage axis; the name rides
# beside it in `card_name` because a table has to print something a reader recognises. A provider
# that exposes no UUID falls back to the name, which is the old behaviour and still better than an
# empty coordinate every card would share.
#
# A HOST WITH NO DEVICE IS A HOST, NOT AN ERROR. The reference this was lifted from refuses to
# stamp a receipt where `nvidia-smi` is absent, which is right for a lab whose every claim reads a
# GPU and wrong for a subsystem that also serves a pure-theory run and this tool's own CI. An
# absent card is recorded as the empty card and the refusal is left to whoever actually needs one.
#
# BUT AN ABSENT CARD AND A BROKEN PROBE ARE NOT THE SAME EMPTY STRING. One says this run measured
# no device and the other says nobody knows what it measured, and a coverage question that cannot
# tell them apart will hand a theory host's cell to a machine whose probe simply fell over. So the
# outcome is recorded beside the value, and a probe that raises is caught HERE and reported rather
# than taking a whole session down, the same translation `mainboard.lab.gates.Gate.evaluate`
# already makes for a precondition check that breaks.
#
# A MIRROR HAS THE SOURCE AND NOT THE HISTORY. A dispatch rsyncs the working tree onto a host and
# the repository stays behind, so `git rev-parse` on a remote card answers nothing and every
# receipt a dispatched job writes could name no commit. The dispatcher declares its own HEAD in
# `MAINBOARD_SOURCE` instead, spelled as `git describe --always --dirty` spells it, and a row
# written under it says `mirrored` so a declared commit is never read as a probed one. Refusing
# outright would say no remote host may ever take a reading, which is the wrong answer to a
# cross-architecture question.

import os
import platform
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..dispatch.shared import git
from ..probe.machine import Machine
from .coverage import Probed

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from pydantic import JsonValue

# The variable a mirrored run reads to learn which source it is, set by whatever dispatched it.
SOURCE_VAR = "MAINBOARD_SOURCE"


class Source(FrozenModel):
    """Which source took a reading, and whether the machine could see the history behind it.

    commit: the short commit the reading was taken at.
    dirty: whether the working tree carried uncommitted changes.
    mirrored: whether the commit was declared by a dispatcher rather than probed from a repo.
    """

    commit: str
    dirty: bool = False
    mirrored: bool = False


def source(repo: Path, *, read: Callable[..., str] = git) -> Source:
    """The commit, the cleanliness and the mirror flag of the tree a reading was taken from.

    repo: the working tree the reading was taken from.
    read: the git reader, the local `git` by default and a stand-in under test.
    """
    head = read("-C", str(repo), "rev-parse", "--short", "HEAD")
    if head:
        return Source(commit=head, dirty=bool(read("-C", str(repo), "status", "--porcelain")))
    declared = os.environ.get(SOURCE_VAR, "")
    if not declared:
        raise RuntimeError(
            f"{repo} is not a git working tree and {SOURCE_VAR} is unset, so this reading could "
            f"name no source; a dispatched job sets {SOURCE_VAR} to the dispatcher's own "
            "`git describe --always --dirty`"
        )
    return Source(
        commit=declared.removesuffix("-dirty"), dirty=declared.endswith("-dirty"), mirrored=True
    )


def installed(name: str) -> str:
    """The installed version of one distribution, `absent` where this environment lacks it."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "absent"


class Card(FrozenModel):
    """The device a reading was taken on, and whether the probe actually found one.

    id: the device UUID, the coverage identity, falling back to the name where none is exposed.
    name: the human name, which is a display column and never an identity.
    driver: the HOST DRIVER version the reading ran under, `610.57.04` shaped.
    runtime: the compute runtime version beside it, `13.3` shaped, the CUDA one on an NVIDIA host.
    capability: the architecture key a kernel dispatches on.
    probed: `found`, `absent` on a host that carries no device, `failed` when the probe broke.
    detail: what the probe said when it broke, empty otherwise.

    THE TWO VERSION FIELDS ARE TWO FACTS AND THE RECEIPT USED TO CARRY ONE OF THEM TWICE. `driver`
    held `cudaDriverGetVersion()`, the maximum CUDA a driver supports, under a name that promised
    the driver, so a generation of receipts stamped `13.3` on a host whose driver is `610.57.04`
    and carried no driver version at all. Three independent reviews on 2026-08-29
    (`fprev_recovery` 7c, `recovery_cost` 7d, `accuracy_selection` 6e) found it separately, which
    is what a field whose name and content disagree costs.
    """

    id: str = ""
    name: str = ""
    driver: str = ""
    runtime: str = ""
    capability: str = ""
    probed: Probed = Probed.ABSENT
    detail: str = ""


def card_of(machine: Machine) -> Card:
    """The first visible device, an empty card where there is none, and why either way.

    A probe is a whole vendor stack behind one attribute, so the set of ways it can fail is not
    ours to enumerate. What matters is that a broken one is REPORTED rather than mistaken for an
    absent device or allowed to take the session down, so the receipt says the machine is unknown
    and a reader can act on that.
    """
    try:
        cards = machine.gpus
    except Exception as error:
        return Card(probed=Probed.FAILED, detail=str(error))
    if not cards:
        return Card(probed=Probed.ABSENT)
    found = cards[0]
    runtime = found.runtime_version
    return Card(
        id=found.uuid or found.label,
        name=found.label,
        driver=found.driver,
        runtime=".".join(str(part) for part in runtime) if runtime else "",
        capability=found.arch_key,
        probed=Probed.FOUND,
    )


def provenance(
    repo: Path,
    *,
    probed: Sequence[str] = (),
    machine: Machine | None = None,
    read: Callable[..., str] = git,
) -> dict[str, JsonValue]:
    """The host, the card and the commit, so no reading can be read on the wrong machine.

    repo: the working tree whose HEAD and cleanliness the reading is stamped with.
    probed: the distributions whose version can move a reading here, recorded on every receipt so
        a row that stops reproducing names its own suspects instead of leaving a reader to guess.
    machine: the probed machine, this host's when omitted.
    read: the git reader, the local `git` by default.
    """
    card = card_of(machine or Machine())
    took = source(repo, read=read)
    return {
        "host": platform.node(),
        "card": card.id,
        "card_probed": str(card.probed),
        "card_name": card.name,
        "card_detail": card.detail,
        "driver": card.driver,
        "runtime": card.runtime,
        "capability": card.capability,
        "commit": took.commit,
        "worktree_dirty": took.dirty,
        "mirrored": took.mirrored,
        "versions": {name: installed(name) for name in probed},
    }
