from mainboard.render.values import columns_of, to_row


def test_to_row_keeps_a_plain_scalar_untouched() -> None:
    assert to_row({"hostname": "gold", "cores": 8, "capped": True, "note": None}) == {
        "hostname": "gold",
        "cores": 8,
        "capped": True,
        "note": None,
    }


def test_to_row_folds_a_nested_mapping_into_one_json_cell() -> None:
    row = to_row({"cgroup": {"limit_bytes": 100, "capped": True}})
    assert row["cgroup"] == '{"limit_bytes": 100, "capped": true}'


def test_to_row_folds_a_list_of_mappings_into_one_json_cell() -> None:
    row = to_row({"gpus": [{"name": "a100"}, {"name": "h100"}]})
    assert row["gpus"] == '[{"name": "a100"}, {"name": "h100"}]'


def test_to_row_folds_a_tuple_the_same_as_a_list() -> None:
    row = to_row({"fabric": ("ib0", "ib1")})
    assert row["fabric"] == '["ib0", "ib1"]'


def test_columns_of_uses_the_given_fields_verbatim() -> None:
    assert columns_of([{"a": 1, "b": 2}], ["b"]) == ["b"]


def test_columns_of_falls_back_to_the_first_row_keys() -> None:
    assert columns_of([{"a": 1}, {"a": 2, "b": 3}], None) == ["a"]


def test_columns_of_is_empty_for_no_rows_and_no_fields() -> None:
    assert columns_of([], None) == []
