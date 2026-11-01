from pathlib import Path

import pytest

from mainboard import ExecutionPlan, Project
from mainboard.dispatch import Dispatcher
from mainboard.manifest import HostProfile

_PLAN = ExecutionPlan(
    host="silo",
    profile=HostProfile(kind="ssh", root="/srv/jobs", platform="linux-64"),
    env="default",
)


def test_a_mirror_with_nothing_to_sync_names_the_manifest_read_and_that_hosts_are_not_inherited(
    workspace: Path,
) -> None:
    nested = workspace / "research" / "nested"
    nested.mkdir(parents=True)
    manifest = Project().manifest(nested)
    manifest.write_text('[workspace]\nname = "nested"\n', encoding="utf-8")

    with pytest.raises(LookupError) as refused:
        Dispatcher(root=nested).mirror(_PLAN, "/srv/jobs")

    assert str(manifest) in str(refused.value)
    assert "does not inherit the [hosts]" in str(refused.value)
    assert "[hosts.silo.sync].include" in str(refused.value)
