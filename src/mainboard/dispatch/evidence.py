# How a run's trial receipts get off the machine that produced them, whichever kind of machine
# that was.
#
# A receipt is one JSON line under the `trial_receipt` key, printed by whatever drove the trial.
# On an ssh host that is enough, because the job script's captured log is a file this workspace
# reads back whole. On a rented instance it is not: a provider hands back a log rather than a
# file, and vast truncates every log line at 500 characters (measured live 2026-08-25), which cuts
# a receipt in half and loses the trial even though the command succeeded and the rental was paid
# for in full. A paid run that produces no evidence is the worst outcome the tool can reach, worse
# than one that fails loudly, because nothing about it says the evidence is missing.
#
# So a provider run writes its receipts to a file the way an ssh job does, and the wrapper frames
# that file back through the one channel a rental is guaranteed to have. The frame is base64 in
# fixed-width chunks, each on its own marked line: base64 because a receipt's own quoting has to
# survive a shell and a log viewer intact, and fixed-width because a chunk is only safe if it is
# narrower than the cut. A block is bounded by its own begin and end lines, and the last complete
# block wins, since vast restarts an exited container and each restart appends another one.
#
# `receipts_in` reads both shapes out of any captured output, the framed block and a plainly
# printed line, so one harvest serves a queued job and a rented instance without asking which it
# is holding.

import base64
import string

# The variable a run reads to learn where to write its receipts, and the file it names. A rented
# machine keeps no workspace, so the path is under `/tmp` rather than derived from a root that
# does not exist there. The shell's own pid is in the name because a cluster node runs several
# jobs at once out of one `/tmp`, and two of them sharing a receipts file would hand each other's
# trials to whichever settled first.
RECEIPTS_VAR = "MAINBOARD_RECEIPTS"
RECEIPTS_FILE = "/tmp/mainboard-receipts.$$.ndjson"

# The key a printed trial receipt carries its payload under, spelled here rather than imported so
# harvesting a receipt never drags the lab machinery in. The same reason `verdicts` spells it.
_RECEIPT = "trial_receipt"

# The frame: one begin line, one end line, and a marker every chunk line carries. Chunks are
# narrower than the tightest provider line limit anyone has measured, with room to spare for the
# marker itself.
_BEGIN = "mainboard-receipts-begin"
_END = "mainboard-receipts-end"
_CHUNK_MARKER = "mainboard-receipt:"
_CHUNK_WIDTH = 240

# What a whole, untorn frame decodes from. Checking the payload against the alphabet and the
# quantum is what lets a torn upload be skipped without asking an exception whether it was well
# formed, so a bad frame costs its own block and never the receipts printed beside it.
_BASE64_ALPHABET = frozenset(string.ascii_letters + string.digits + "+/=")


def staging() -> str:
    """The shell that points a run at its receipts file and starts that file empty.

    Emptying matters on a provider that restarts an exited container, since the second run would
    otherwise frame the first run's receipts back alongside its own.
    """
    return f"export {RECEIPTS_VAR}={RECEIPTS_FILE}; : > ${RECEIPTS_VAR}"


def framing() -> str:
    """The shell that emits the receipts file back through the run's captured output.

    One pipeline of tools any Linux image already carries, since this runs on a rented machine
    that this workspace never provisioned and cannot install anything on. A run that wrote no
    receipts emits no block at all, so an ordinary command's log is left exactly as it was.
    """
    file = f'"${RECEIPTS_VAR}"'
    return (
        f"if [ -s {file} ]; then echo {_BEGIN}; "
        f'base64 < {file} | tr -d "\\n" | fold -w {_CHUNK_WIDTH} '
        f"| sed 's/^/{_CHUNK_MARKER}/'; echo {_END}; fi"
    )


def unframed(log: str) -> str:
    """The receipts text `log`'s last whole frame carries, empty when it holds none.

    The last block rather than the first, since a provider that restarts an exited container
    appends one block per run and the newest is the one this handle's verdict is about. A block
    cut off mid-upload is skipped for the next one down, so a torn log costs its own frame and
    nothing else.

    log: the run's captured output, as its backend handed it over.
    """
    lines = [line.strip() for line in log.splitlines()]
    for start in reversed([at for at, line in enumerate(lines) if line == _BEGIN]):
        ends = [at for at, line in enumerate(lines[start:], start) if line == _END]
        if not ends:
            continue
        payload = "".join(
            line.removeprefix(_CHUNK_MARKER)
            for line in lines[start + 1 : ends[0]]
            if line.startswith(_CHUNK_MARKER)
        )
        if payload and len(payload) % 4 == 0 and set(payload) <= _BASE64_ALPHABET:
            return base64.b64decode(payload).decode(errors="replace")
    return ""


def receipts_in(log: str) -> tuple[str, ...]:
    """Every trial receipt `log` carries, framed or printed plainly, in first-seen order.

    One harvest for both worlds. A queued job prints its receipts straight into a log file this
    workspace reads back whole, and a rented instance frames them through that same output because
    it has nowhere else to put them, so a caller settling a run never has to know which of the two
    it is holding. Duplicates collapse, since a harness that both writes and prints a receipt has
    still only run the one trial.

    log: the run's captured output, as its backend handed it over.
    """
    lines = [*unframed(log).splitlines(), *log.splitlines()]
    found = [line.strip() for line in lines if _RECEIPT in line and line.strip()]
    return tuple(dict.fromkeys(found))
