import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.manifest import HostProfile, Manifest, Observe, QueuePolicy, Sync

from ..strategies import PATHS, WORDS

# Sync rules generate unique so a list is its own union, which is what lets the identity law
# below compare a merged profile against the one that went in.
_RULES = st.lists(PATHS, max_size=3, unique=True)
_PROFILES = st.builds(
    HostProfile,
    kind=WORDS,
    modules=st.dictionaries(WORDS, WORDS, max_size=2),
    vars=st.dictionaries(WORDS, WORDS, max_size=2),
    queues=st.dictionaries(
        WORDS, st.builds(QueuePolicy, max_walltime=st.sampled_from(["", "01:00:00"])), max_size=2
    ),
    sync=st.builds(Sync, include=_RULES, exclude=_RULES, protect=_RULES),
)


@given(base=_PROFILES, child=_PROFILES)
def test_inheriting_never_drops_a_rule_and_an_empty_base_is_the_identity(
    base: HostProfile, child: HostProfile
) -> None:
    """A host adding one exclude used to silently drop the workspace-wide protect rules."""
    landed = child.inheriting(base)
    assert set(base.sync.protect) <= set(landed.sync.protect)
    assert set(child.sync.protect) <= set(landed.sync.protect)
    assert landed.sync.exclude == list(dict.fromkeys([*base.sync.exclude, *child.sync.exclude]))
    assert landed.modules == {**base.modules, **child.modules}
    assert landed.vars == {**base.vars, **child.vars}
    assert landed.queues == {**base.queues, **child.queues}
    assert child.inheriting(HostProfile()) == child
    assert HostProfile().inheriting(base) == base


def test_profiles_resolve_from_the_manifest_and_a_stranger_still_gets_the_defaults(
    loaded: Manifest,
) -> None:
    """Declared hosts layer over `[hosts.defaults]`, and an undeclared one is the base alone."""
    gold = loaded.profile("gold")
    assert gold.kind == "ssh"
    assert gold.env == "serving"
    assert gold.sync.include == ["packages"]
    assert gold.defaults.walltime == "00:30:00"
    assert gold.observe == Observe()
    miyabi = loaded.profile("miyabi-g")
    assert miyabi.sync.include == ["packages"]
    assert miyabi.sync.exclude == ["data/raw"]
    assert miyabi.sync.protect == ["results/***"]
    stranger = loaded.profile("stranger")
    assert stranger.kind == "auto"
    assert stranger.sync.protect == ["results/***"]


@pytest.mark.parametrize(
    ("queue", "walltime", "admitted"),
    [
        ("short-g", "06:00:00", True),
        ("short-g", "08:00:00", False),
        ("unknown-queue", "99:00:00", True),
    ],
)
def test_a_queue_admits_only_a_walltime_under_its_declared_ceiling(
    loaded: Manifest, queue: str, walltime: str, admitted: bool
) -> None:
    """A queue the host never declared is permissive, since the ceiling is what it declares."""
    assert loaded.profile("miyabi-g").policy(queue).admits_walltime(walltime) is admitted


def test_a_hosts_own_observe_posture_wins_wholesale_over_the_defaults_profile() -> None:
    """The posture is one table, so naming `level` does not drag the base's channel along."""
    base = HostProfile(observe=Observe(level="stream", channel="stream", poll_seconds=5.0))
    own = HostProfile(observe=Observe(level="off"))
    assert own.inheriting(base).observe == Observe(level="off")
    assert HostProfile().inheriting(base).observe == base.observe
    assert HostProfile().observe == Observe(level="poll", channel="auto", poll_seconds=30.0)
