from mainboard.dispatch.mirroring.sync import outermost


def test_a_path_under_another_root_is_listed_once() -> None:
    roots = [
        "mb.toml",
        "research/x/cutoken",
        "research/x/cutoken/pyproject.toml",
        "pkg/pyproject.toml",
    ]
    assert outermost(roots) == ["mb.toml", "research/x/cutoken", "pkg/pyproject.toml"]
    assert outermost([".", "a/b"]) == ["."]
