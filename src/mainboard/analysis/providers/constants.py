"""The constants of `providers` other modules share."""

_SHADOWED = "shadowed"

_PLACE = {
    "fact_id": "VARCHAR",
    "path": "VARCHAR",
    "language": "VARCHAR",
    "start_line": "BIGINT",
    "start_column": "BIGINT",
}

SYNTAX_TABLES = {
    "facts": _PLACE | {"parser_gaps": "BIGINT", "parser_message": "VARCHAR"},
    "imports": _PLACE | {"name": "VARCHAR"},
    "conditions": _PLACE | {"condition": "VARCHAR"},
    "conformances": _PLACE
    | {
        "owner_id": "BIGINT",
        "owner": "VARCHAR",
        "owner_kind": "VARCHAR",
        "inherited": "VARCHAR",
        _SHADOWED: "BIGINT",
    },
    "attributes": _PLACE
    | {
        "owner_id": "BIGINT",
        "owner": "VARCHAR",
        "attribute": "VARCHAR",
        "container_id": "BIGINT",
        "container": "VARCHAR",
        "container_kind": "VARCHAR",
        "is_static": "BIGINT",
        _SHADOWED: "BIGINT",
    },
    "attribute_options": _PLACE
    | {
        "owner_id": "BIGINT",
        "owner": "VARCHAR",
        "container_id": "BIGINT",
        "container": "VARCHAR",
        "container_kind": "VARCHAR",
        "is_static": "BIGINT",
        "attribute": "VARCHAR",
        "attribute_shadowed": "BIGINT",
        "option": "VARCHAR",
        "receiver": "VARCHAR",
        _SHADOWED: "BIGINT",
    },
    "calls": _PLACE | {"name": "VARCHAR", "receiver": "VARCHAR", _SHADOWED: "BIGINT"},
    "references": _PLACE | {"name": "VARCHAR", "receiver": "VARCHAR", _SHADOWED: "BIGINT"},
    "states": _PLACE | {"name": "VARCHAR", "attribute": "VARCHAR", "is_private": "BIGINT"},
    "erasures": _PLACE | {"view_name": "VARCHAR"},
}

TABLES = SYNTAX_TABLES | {
    "toolchain": {
        "compiler": "VARCHAR",
        "compiler_version": "VARCHAR",
        "sdk": "VARCHAR",
        "host_libraries": "VARCHAR",
        "helper_identity": "VARCHAR",
        "helper_source_sha256": "VARCHAR",
    },
}
