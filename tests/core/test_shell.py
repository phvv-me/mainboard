import pytest

from mainboard import script, sh


def test_sh_quotes_every_interpolation() -> None:
    root = "/work/my projects"
    command = "echo $(rm -rf /)"
    line = sh(t"cd {root} && {command}")
    assert line == "cd '/work/my projects' && 'echo $(rm -rf /)'"


def test_sh_leaves_static_text_alone() -> None:
    env = "serving"
    assert sh(t"mainboard run --env {env} -- true") == "mainboard run --env serving -- true"


def test_sh_refuses_plain_strings() -> None:
    with pytest.raises(TypeError):
        sh("cd /tmp && rm -rf *")  # type: ignore[arg-type]  # the refusal under test


def test_script_lands_trusted_fragments_verbatim() -> None:
    inner = sh(t"echo {'a b'}")
    line = script(t"bash -lc {inner}")
    assert line == "bash -lc echo 'a b'"
