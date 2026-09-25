import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path, PurePath

import pytest

from mainboard.testing import POSIX_TOOLS, Spelling, WindowsLike, case_variants, first_party

_POSIX = pytest.mark.skipif(sys.platform == "win32", reason="builds the POSIX link farm")


def _program(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    made = directory / name
    made.write_text("#!/bin/sh\n", encoding="utf-8")
    made.chmod(0o755)
    return made


@_POSIX
def test_the_posix_path_keeps_every_program_but_the_ones_windows_lacks(tmp_path: Path) -> None:
    """Links, not directories, since `/usr/bin` holds git and bash side by side."""
    bin_dir = tmp_path / "bin"
    for name in ("git", "bash", "rsync"):
        _program(bin_dir, name)
    (bin_dir / "notes.txt").write_text("", encoding="utf-8")
    inherited = os.pathsep.join([str(bin_dir), str(tmp_path / "missing"), ""])
    like = WindowsLike(tmp_path / "farms", platform="darwin")

    pruned = like.path(inherited)

    assert sorted(path.name for path in Path(pruned).iterdir()) == ["git"]
    assert like.path(inherited) == pruned


def test_on_windows_the_directories_that_bring_posix_tools_are_dropped(tmp_path: Path) -> None:
    system, git_usr = tmp_path / "System32", tmp_path / "Git" / "usr" / "bin"
    _program(system, "where.exe")
    _program(git_usr, "bash.exe")
    like = WindowsLike(tmp_path / "farms", platform="win32")
    assert like.path(os.pathsep.join([str(system), str(git_usr)])) == os.fspath(system)


def test_the_projects_own_code_is_first_party_and_the_interpreters_is_not() -> None:
    assert first_party(__file__)
    assert not first_party(os.path.join(sysconfig.get_path("stdlib"), "os.py"))
    assert not first_party("<frozen posixpath>")


@pytest.mark.windows_like
def test_a_marked_test_reads_paths_and_tools_as_windows_would(tmp_path: Path) -> None:
    """Text is backslashed; the path, its parse and the system all still reach the same file."""
    if sys.platform != "win32":
        assert all(shutil.which(tool) is None for tool in ("bash", "rsync"))
    target = tmp_path / "pkg" / "one.py"
    target.parent.mkdir()
    target.write_text("x = 1\n", encoding="utf-8")
    spelled = str(target)

    assert "/" not in spelled
    assert f"{target.relative_to(tmp_path)}" == "pkg\\one.py"
    assert os.fspath(target).replace("\\", "/") == target.as_posix()
    assert Path(spelled) == target
    with open(spelled, encoding="utf-8") as handle:
        assert handle.read() == "x = 1\n"
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, sys; print(os.getcwd(), sys.argv[1:])",
            spelled,
            "a\\b",
        ],
        cwd=str(tmp_path),
        env={**os.environ, "HOME_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    where, argv = done.stdout.split(" ", 1)
    assert Path(where).resolve() == tmp_path.resolve()
    assert argv.strip() == repr([os.fspath(target), "a\\b"])
    probe = subprocess.list2cmdline([sys.executable, "-c", "pass", spelled])
    assert (
        subprocess.run(probe if os.name == "nt" else f"test -f {spelled}", shell=True).returncode
        == 0
    )
    assert subprocess.run(Path(sys.executable), input=b"", check=True).returncode == 0


@pytest.mark.windows_like
def test_a_live_temporary_file_is_held_and_a_link_needs_a_privilege(tmp_path: Path) -> None:
    """The two sharing rules POSIX never enforces, which each broke a Windows run."""
    with (
        tempfile.NamedTemporaryFile(dir=tmp_path) as held,
        pytest.raises(PermissionError, match="another process"),
    ):
        Path(held.name).read_bytes()
    with pytest.raises(FileNotFoundError):
        Path(held.name).read_bytes()
    with tempfile.NamedTemporaryFile(dir=tmp_path, delete=False) as kept:
        Path(kept.name).read_bytes()
    descriptor = os.open(kept.name, os.O_RDONLY)
    with open(descriptor, "rb") as reopened:
        assert reopened.read() == b""
    with pytest.raises(OSError, match="privilege is not held"):
        (tmp_path / "link").symlink_to(kept.name)


def test_a_rendered_spelling_is_read_back_only_where_the_system_reads_it() -> None:
    spelling = Spelling()
    windows = spelling.text(PurePath.__str__)
    absolute, relative = PurePath("/w/pkg"), PurePath("pkg/one.py")

    assert windows(absolute) == "\\w\\pkg"
    assert windows(relative) == "pkg\\one.py"
    assert spelling.real("cd \\w\\pkg && ls pkg\\one.py") == "cd /w/pkg && ls pkg/one.py"
    assert spelling.real("ls pkg\\one.py \\w\\pkg", relative=False) == "ls pkg\\one.py /w/pkg"
    assert spelling.real("plain") == "plain"


def test_the_whole_session_can_run_as_windows(pytester: pytest.Pytester) -> None:
    pytester.makeconftest('pytest_plugins = ["mainboard.testing"]')
    pytester.makepyfile(
        """
        from pathlib import Path

        def test_backslashed():
            assert str(Path("a/b")) == "a\\\\b"
        """
    )
    result = pytester.runpytest("-p", "no:cacheprovider", "-o", "addopts=", "--windows-like")
    result.assert_outcomes(passed=1)


def test_the_flavor_and_newline_fixtures_cover_both_of_each(
    path_flavor: type[PurePath], newline: str
) -> None:
    assert path_flavor("a", "b").as_posix() == "a/b"
    assert f"one{newline}two{newline}".splitlines() == ["one", "two"]


def test_case_variants_are_the_spellings_a_case_insensitive_filesystem_merges() -> None:
    assert case_variants("README.md") == ("README.md", "readme.md", "README.MD", "readme.MD")
    assert case_variants("abc") == ("abc", "ABC")
    assert {"bash", "rsync", "flock", "timeout"} <= POSIX_TOOLS
