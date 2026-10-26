import mmap
import re
from collections.abc import Iterable
from pathlib import Path
from struct import error, iter_unpack, unpack_from

from patos import FrozenModel

# The dynamic-section tags a dynamic loader acts on, and the section types that locate them.
_DT_NEEDED, _DT_SONAME, _DT_RPATH, _DT_RUNPATH = 1, 14, 15, 29
_SHT_DYNAMIC = 6
_SECTION = "<IIQQQQII"
_ORIGIN = re.compile(r"\$\{?ORIGIN\}?")


class Elf(FrozenModel):
    """What the dynamic loader reads from one shared library: who it needs and where it looks.

    Parsed from the section headers of a 64-bit little-endian ELF file, which is every library
    the CUDA families ship on Linux; any other file (a linker script, a macOS dylib) answers
    empty, since it names nothing the loader would act on.

    needed: the sonames the library links against (`DT_NEEDED`).
    rpath: `DT_RPATH`, searched before `LD_LIBRARY_PATH` when no runpath is set.
    runpath: `DT_RUNPATH`, searched after it. Both have `$ORIGIN` already expanded.
    """

    path: Path
    soname: str = ""
    needed: tuple[str, ...] = ()
    rpath: tuple[Path, ...] = ()
    runpath: tuple[Path, ...] = ()

    @classmethod
    def read(cls, path: Path) -> Elf:
        """The dynamic section of `path`, only its headers read however large the library is."""
        try:
            tags = Elf._dynamic(path)
        except OSError, error, ValueError:
            return cls(path=path)
        return cls(
            path=path,
            soname=next(iter(tags.get(_DT_SONAME, ())), ""),
            needed=tags.get(_DT_NEEDED, ()),
            rpath=cls._directories(path, tags.get(_DT_RPATH, ())),
            runpath=cls._directories(path, tags.get(_DT_RUNPATH, ())),
        )

    @staticmethod
    def _directories(path: Path, entries: Iterable[str]) -> tuple[Path, ...]:
        """The directories of colon-separated `entries`, `$ORIGIN` replaced by `path`'s own."""
        return tuple(
            Path(_ORIGIN.sub(str(path.parent), directory))
            for entry in entries
            for directory in entry.split(":")
            if directory
        )

    @staticmethod
    def _dynamic(path: Path) -> dict[int, tuple[str, ...]]:
        """Every string-valued tag of `path`'s dynamic section, by tag number."""
        with path.open("rb") as file:
            header = file.read(64)
            if header[:4] != b"\x7fELF" or header[4:6] != b"\x02\x01":
                raise ValueError(path)
            (table,) = unpack_from("<Q", header, 0x28)
            size, count = unpack_from("<HH", header, 0x3A)
            file.seek(table)
            raw = file.read(size * count)
            sections = [unpack_from(_SECTION, raw, at * size) for at in range(count)]
            dynamic = next(section for section in sections if section[1] == _SHT_DYNAMIC)
            file.seek(dynamic[4])
            entries = file.read(dynamic[5])
            names = sections[dynamic[6]]
            file.seek(names[4])
            strings = file.read(names[5])
        tags: dict[int, list[str]] = {}
        for tag, value in iter_unpack("<qQ", entries):
            if tag in {_DT_NEEDED, _DT_SONAME, _DT_RPATH, _DT_RUNPATH}:
                tags.setdefault(tag, []).append(
                    strings[value : strings.index(b"\0", value)].decode()
                )
        return {tag: tuple(values) for tag, values in tags.items()}

    def mentions(self, stem: str) -> tuple[str, ...]:
        """The library names beginning `stem` that the file's bytes spell out, as it spells them.

        A library that opens its siblings by name (cuDNN's dispatcher and its graph library)
        lists none of them as needed, so the strings it carries are the only record. They often
        end at `.so`, the version being appended when the call is made.
        """
        # Opening on the literal stem, with no `\b` before it, is what lets the regex engine skip
        # to each occurrence: with the assertion it tested every byte, 25x slower over gigabytes.
        pattern = re.compile(re.escape(stem.encode()) + rb"[\w+-]*\.so(?:\.\d+)*")
        with (
            self.path.open("rb") as file,
            mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data,
        ):
            return tuple(sorted({found.decode() for found in pattern.findall(data)}))
