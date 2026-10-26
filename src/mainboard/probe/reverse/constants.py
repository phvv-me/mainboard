"""The constants of `reverse` other modules share."""

# These standard toolkit spellings are checked against the installed tool's --help before
# analysis. TODO(lead): exercise these contracts on real CUDA 13.x binaries and library fixtures.
STANDARD_FLAGS = {
    "cuobjdump": (
        "--list-elf",
        "--list-ptx",
        "--dump-sass",
        "--function",
        "--dump-resource-usage",
        "--extract-elf",
    ),
    # -c is --print-code, not a CFG output flag; this provider parses SASS text only.
    "nvdisasm": ("-c",),
}
