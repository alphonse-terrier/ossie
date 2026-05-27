"""Tests for the bidirectional OSI ↔ Honeydew converter."""

import json
import os
import sys
import warnings
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from honeydew_osi_converter import (
    HoneydewConversionError,
    _assign_metrics_to_entities,
    _build_osi_metadata,
    _check_safe_path,
    _fields_to_honeydew,
    _find_entity_in_expression,
    _honeydew_datatype_to_osi_dimension,
    _is_simple_identifier,
    _osi_field_to_honeydew_datatype,
    _parse_osi_source,
    _pick_ansi_expression,
    _read_osi_metadata,
    convert_honeydew_to_osi,
    convert_osi_to_honeydew,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

OSI_VERSION = "0.2.0.dev0"


def _osi(model_dict):
    return yaml.dump(
        {"version": OSI_VERSION, "semantic_model": [model_dict]},
        default_flow_style=False,
        sort_keys=False,
    )


def _minimal_osi_field(name, expr, is_dimension=True, is_time=False):
    field = {
        "name": name,
        "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": expr}]},
    }
    if is_dimension:
        field["dimension"] = {"is_time": is_time}
    return field


def _minimal_model():
    return {
        "name": "test_model",
        "datasets": [
            {
                "name": "orders",
                "source": "db.schema.orders",
                "primary_key": ["order_id"],
                "fields": [
                    _minimal_osi_field("order_id", "order_id"),
                    _minimal_osi_field("order_date", "order_date", is_time=True),
                    _minimal_osi_field("total", "total_amount", is_dimension=False),
                ],
            }
        ],
    }


def _write_workspace(tmp_dir, workspace_name, entities):
    """Write a minimal Honeydew workspace to tmp_dir."""
    workspace_path = os.path.join(tmp_dir, "workspace.yml")
    with open(workspace_path, "w") as f:
        yaml.dump({"type": "workspace", "name": workspace_name}, f)

    for e in entities:
        ename = e["name"]
        base = os.path.join(tmp_dir, "schema", ename)
        os.makedirs(os.path.join(base, "datasets"), exist_ok=True)
        os.makedirs(os.path.join(base, "attributes"), exist_ok=True)
        os.makedirs(os.path.join(base, "metrics"), exist_ok=True)

        entity_dict = {
            "type": "entity",
            "name": ename,
            "keys": e.get("keys", []),
            "key_dataset": e.get("key_dataset", ename),
            "relations": e.get("relations", []),
        }
        for k in ("owner", "display_name", "hidden", "folder"):
            if k in e:
                entity_dict[k] = e[k]
        with open(os.path.join(base, f"{ename}.yml"), "w") as f:
            yaml.dump(entity_dict, f)

        ds_name = e.get("key_dataset", ename)
        ds_dict = {
            "type": "dataset",
            "entity": ename,
            "name": ds_name,
            "sql": e.get("sql", "DB.SCHEMA." + ename.upper()),
            "dataset_type": "table",
            "attributes": e.get("dataset_attrs", []),
        }
        with open(os.path.join(base, "datasets", f"{ds_name}.yml"), "w") as f:
            yaml.dump(ds_dict, f)

        for attr in e.get("calc_attrs", []):
            with open(os.path.join(base, "attributes", f"{attr['name']}.yml"), "w") as f:
                yaml.dump(attr, f)

        for m in e.get("metrics", []):
            with open(os.path.join(base, "metrics", f"{m['name']}.yml"), "w") as f:
                yaml.dump(m, f)


def _osi_roundtrip(model_dict, tmp_path):
    """OSI → Honeydew → OSI; returns the semantic model dict."""
    files = convert_osi_to_honeydew(_osi(model_dict))
    for rel_path, content in files.items():
        p = tmp_path / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))["semantic_model"][0]


def _honeydew_roundtrip(entities, tmp_path):
    """Honeydew → OSI → Honeydew; returns Path to the output workspace directory."""
    _write_workspace(str(tmp_path), "ws", entities)
    osi_yaml = convert_honeydew_to_osi(str(tmp_path))
    files = convert_osi_to_honeydew(osi_yaml)
    out_dir = tmp_path / "out"
    for rel_path, content in files.items():
        p = out_dir / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests – helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("expr,expected", [
    ("order_id", True),
    ("SUM(x)", False),
    ("orders.id", False),
    ("1col", False),
    ("_hidden", True),
])
def test_is_simple_identifier(expr, expected):
    assert _is_simple_identifier(expr) is expected


@pytest.mark.parametrize("source,expected_sql,expected_type", [
    ("db.schema.table", "db.schema.table", "table"),
    ("SELECT id FROM foo", "SELECT id FROM foo", "sql"),
    ("WITH cte AS (SELECT 1) SELECT * FROM cte", "WITH cte AS (SELECT 1) SELECT * FROM cte", "sql"),
    ("", "", "table"),
])
def test_parse_osi_source(source, expected_sql, expected_type):
    sql, dtype = _parse_osi_source(source)
    assert sql == expected_sql and dtype == expected_type


@pytest.mark.parametrize("field,expected_dt", [
    ({"dimension": {"is_time": True}}, "timestamp"),
    ({"dimension": {"is_time": False}}, "string"),
    ({}, "number"),
])
def test_osi_field_to_honeydew_datatype(field, expected_dt):
    assert _osi_field_to_honeydew_datatype(field) == expected_dt


@pytest.mark.parametrize("datatype,expected_dim", [
    ("date", {"is_time": True}),
    ("timestamp", {"is_time": True}),
    ("string", {"is_time": False}),
    ("bool", {"is_time": False}),
    ("number", None),
    ("float", None),
])
def test_honeydew_datatype_to_osi_dimension(datatype, expected_dim):
    assert _honeydew_datatype_to_osi_dimension(datatype) == expected_dim


@pytest.mark.parametrize("expr,entities,expected", [
    ("SUM(orders.total)", {"orders", "customers"}, "orders"),
    ("orders.a / customers.b", {"orders", "customers"}, "orders"),
    ("COUNT(*)", {"orders"}, None),
    ("SUM(foo.col)", {"orders"}, None),
])
def test_find_entity_in_expression(expr, entities, expected):
    assert _find_entity_in_expression(expr, entities) == expected


def test_pick_ansi_expression_ansi_preferred():
    expr = {"dialects": [
        {"dialect": "SNOWFLAKE", "expression": "col::VARCHAR"},
        {"dialect": "ANSI_SQL", "expression": "col"},
    ]}
    assert _pick_ansi_expression(expr, "f") == "col"


def test_pick_ansi_expression_fallback_warns():
    expr = {"dialects": [{"dialect": "SNOWFLAKE", "expression": "col::VARCHAR"}]}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _pick_ansi_expression(expr, "f")
    assert result == "col::VARCHAR"
    assert any("ANSI_SQL" in str(x.message) for x in w)


@pytest.mark.parametrize("expression", [None, {"dialects": []}])
def test_pick_ansi_expression_returns_none(expression):
    assert _pick_ansi_expression(expression, "f") is None


def test_pick_ansi_expression_non_dict_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _pick_ansi_expression("just_a_string", "f")
    assert result is None
    assert any("must be a mapping" in str(x.message) for x in w)


# ─────────────────────────────────────────────────────────────────────────────
# OSI metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_build_and_read_ai_context_string():
    section = _build_osi_metadata(ai_context="orders, purchases")
    result = _read_osi_metadata({"metadata": [section]})
    assert result["ai_context"] == "orders, purchases"


def test_build_and_read_ai_context_dict():
    ctx = {"instructions": "Use for sales", "synonyms": ["orders", "purchases"]}
    section = _build_osi_metadata(ai_context=ctx)
    result = _read_osi_metadata({"metadata": [section]})
    assert result["ai_context"] == ctx


def test_build_and_read_unique_keys():
    uks = [["col1", "col2"], ["col3"]]
    section = _build_osi_metadata(unique_keys=uks)
    result = _read_osi_metadata({"metadata": [section]})
    assert result["unique_keys"] == uks


def test_build_and_read_custom_extensions():
    exts = [{"vendor_name": "SNOWFLAKE", "data": '{"warehouse": "WH"}'}]
    section = _build_osi_metadata(custom_extensions=exts)
    result = _read_osi_metadata({"metadata": [section]})
    assert result["custom_extensions"] == exts


def test_read_osi_metadata_no_osi_section():
    assert _read_osi_metadata({"metadata": [{"name": "other", "metadata": []}]}) == {}


def test_read_osi_metadata_no_metadata():
    assert _read_osi_metadata({}) == {}


def test_build_osi_metadata_nothing_to_store():
    assert _build_osi_metadata() is None


# ─────────────────────────────────────────────────────────────────────────────
# Assign metrics to entities
# ─────────────────────────────────────────────────────────────────────────────

def test_assign_metrics_by_expression():
    metrics = [{"name": "total", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(orders.total)"}]}}]
    result = _assign_metrics_to_entities(metrics, ["orders", "customers"])
    assert "total" in [m["name"] for m in result.get("orders", [])]


def test_assign_metrics_honeydew_hint_takes_priority():
    metrics = [{
        "name": "cnt",
        "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(orders.x)"}]},
        "custom_extensions": [{"vendor_name": "HONEYDEW", "data": '{"entity": "customers"}'}],
    }]
    result = _assign_metrics_to_entities(metrics, ["orders", "customers"])
    assert "cnt" in [m["name"] for m in result.get("customers", [])]
    assert "orders" not in result


def test_assign_metrics_fallback_to_first_entity():
    metrics = [{"name": "cnt", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "COUNT(*)"}]}}]
    with warnings.catch_warnings(record=True):
        result = _assign_metrics_to_entities(metrics, ["orders"])
    assert "cnt" in [m["name"] for m in result.get("orders", [])]


def test_assign_metrics_no_entities():
    metrics = [{"name": "m", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "COUNT(*)"}]}}]
    with warnings.catch_warnings(record=True):
        result = _assign_metrics_to_entities(metrics, [])
    assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# Path traversal guard
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rel_path,expected", [
    ("workspace.yml", True),
    ("schema/orders/orders.yml", True),
    ("schema/orders/datasets/orders.yml", True),
    ("../evil.yml", False),
    ("../../etc/passwd", False),
    ("schema/../../../evil", False),
])
def test_check_safe_path(rel_path, expected):
    output_abs = os.path.abspath("/tmp/test_output")
    assert _check_safe_path(output_abs, rel_path) is expected


# ─────────────────────────────────────────────────────────────────────────────
# OSI → Honeydew integration tests
# ─────────────────────────────────────────────────────────────────────────────

def test_osi_to_honeydew_workspace_yml():
    files = convert_osi_to_honeydew(_osi(_minimal_model()))
    ws = yaml.safe_load(files["workspace.yml"])
    assert ws["name"] == "test_model" and ws["type"] == "workspace"


def test_osi_to_honeydew_entity_yml():
    files = convert_osi_to_honeydew(_osi(_minimal_model()))
    entity = yaml.safe_load(files["schema/orders/orders.yml"])
    assert entity["name"] == "orders"
    assert entity["keys"] == ["order_id"]
    assert entity["key_dataset"] == "orders"


def test_osi_to_honeydew_dataset_yml():
    files = convert_osi_to_honeydew(_osi(_minimal_model()))
    ds = yaml.safe_load(files["schema/orders/datasets/orders.yml"])
    assert ds["sql"] == "db.schema.orders"
    assert ds["dataset_type"] == "table"


def test_osi_to_honeydew_simple_fields_become_dataset_attributes():
    files = convert_osi_to_honeydew(_osi(_minimal_model()))
    ds = yaml.safe_load(files["schema/orders/datasets/orders.yml"])
    names = [a["name"] for a in ds["attributes"]]
    assert "order_id" in names and "order_date" in names and "total" in names


@pytest.mark.parametrize("field_name,expected_dt", [
    ("order_date", "timestamp"),
    ("total", "number"),
])
def test_osi_to_honeydew_field_datatypes(field_name, expected_dt):
    files = convert_osi_to_honeydew(_osi(_minimal_model()))
    ds = yaml.safe_load(files["schema/orders/datasets/orders.yml"])
    attrs = {a["name"]: a for a in ds["attributes"]}
    assert attrs[field_name]["datatype"] == expected_dt


def test_osi_to_honeydew_complex_expression_becomes_calculated_attribute():
    model = {"name": "m", "datasets": [{"name": "orders", "source": "db.s.orders", "fields": [{
        "name": "disc_price",
        "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "price * (1 - discount)"}]},
        "dimension": {"is_time": False},
    }]}]}
    files = convert_osi_to_honeydew(_osi(model))
    assert "schema/orders/attributes/disc_price.yml" in files
    calc = yaml.safe_load(files["schema/orders/attributes/disc_price.yml"])
    assert calc["type"] == "calculated_attribute"
    assert calc["sql"] == "price * (1 - discount)"


def test_osi_to_honeydew_label_mapped_to_honeydew_labels():
    model = {"name": "m", "datasets": [{"name": "orders", "source": "db.s.orders", "fields": [{
        "name": "status",
        "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "status"}]},
        "dimension": {"is_time": False},
        "label": "sales",
    }]}]}
    files = convert_osi_to_honeydew(_osi(model))
    ds = yaml.safe_load(files["schema/orders/datasets/orders.yml"])
    attrs = {a["name"]: a for a in ds["attributes"]}
    assert "sales" in attrs["status"]["labels"]


def test_osi_to_honeydew_ai_context_string_merged_into_description():
    model = {"name": "m", "datasets": [{"name": "orders", "source": "db.s.orders", "fields": [{
        "name": "total",
        "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "total"}]},
        "description": "Base desc",
        "ai_context": "revenue, earnings",
    }]}]}
    files = convert_osi_to_honeydew(_osi(model))
    ds = yaml.safe_load(files["schema/orders/datasets/orders.yml"])
    attrs = {a["name"]: a for a in ds["attributes"]}
    assert "revenue, earnings" in attrs["total"]["description"]


def test_osi_to_honeydew_ai_context_dict_instructions_merged_into_description():
    model = {"name": "m", "datasets": [{"name": "orders", "source": "db.s.orders", "fields": [{
        "name": "total",
        "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "total"}]},
        "ai_context": {"instructions": "Use for revenue", "synonyms": ["rev", "earnings"]},
    }]}]}
    files = convert_osi_to_honeydew(_osi(model))
    ds = yaml.safe_load(files["schema/orders/datasets/orders.yml"])
    attrs = {a["name"]: a for a in ds["attributes"]}
    assert "Use for revenue" in attrs["total"]["description"]
    assert "rev" in attrs["total"]["labels"]


def test_osi_to_honeydew_ai_context_dict_stored_in_metadata():
    model = {"name": "m", "datasets": [{"name": "orders", "source": "db.s.orders", "fields": [{
        "name": "total",
        "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "total"}]},
        "ai_context": {"instructions": "Use for revenue", "synonyms": ["rev"]},
    }]}]}
    files = convert_osi_to_honeydew(_osi(model))
    ds = yaml.safe_load(files["schema/orders/datasets/orders.yml"])
    attr = next(a for a in ds["attributes"] if a["name"] == "total")
    osi_section = next((s for s in attr.get("metadata", []) if s["name"] == "osi"), None)
    assert osi_section is not None
    assert any(i["name"] == "ai_context" for i in osi_section["metadata"])


def test_osi_to_honeydew_unique_keys_stored_in_entity_metadata():
    model = {"name": "m", "datasets": [{"name": "items", "source": "db.s.items",
        "primary_key": ["item_id"],
        "unique_keys": [["sku"], ["item_id", "variant"]],
        "fields": []}]}
    files = convert_osi_to_honeydew(_osi(model))
    entity = yaml.safe_load(files["schema/items/items.yml"])
    osi_section = next((s for s in entity.get("metadata", []) if s["name"] == "osi"), None)
    assert osi_section is not None
    uk_item = next((i for i in osi_section["metadata"] if i["name"] == "unique_keys"), None)
    assert uk_item is not None
    assert json.loads(uk_item["value"]) == [["sku"], ["item_id", "variant"]]


def test_osi_to_honeydew_non_honeydew_extensions_stored_in_metadata():
    model = {"name": "m", "datasets": [{"name": "orders", "source": "db.s.orders",
        "custom_extensions": [{"vendor_name": "SNOWFLAKE", "data": '{"warehouse": "WH"}'}],
        "fields": []}]}
    files = convert_osi_to_honeydew(_osi(model))
    entity = yaml.safe_load(files["schema/orders/orders.yml"])
    osi_section = next((s for s in entity.get("metadata", []) if s["name"] == "osi"), None)
    assert osi_section is not None
    ext_item = next((i for i in osi_section["metadata"] if i["name"] == "custom_extensions"), None)
    assert ext_item is not None
    exts = json.loads(ext_item["value"])
    assert any(e["vendor_name"] == "SNOWFLAKE" for e in exts)


def test_osi_to_honeydew_relationship_name_stored_in_relation():
    model = {"name": "m", "datasets": [
        {"name": "orders", "source": "db.s.orders", "fields": []},
        {"name": "customers", "source": "db.s.customers", "fields": []},
    ], "relationships": [{"name": "orders_to_customers", "from": "orders", "to": "customers",
        "from_columns": ["cid"], "to_columns": ["id"]}]}
    files = convert_osi_to_honeydew(_osi(model))
    entity = yaml.safe_load(files["schema/orders/orders.yml"])
    assert entity["relations"][0]["name"] == "orders_to_customers"


def test_osi_to_honeydew_model_ai_context_stored_in_workspace_metadata():
    model = {"name": "m", "datasets": [],
        "ai_context": {"instructions": "Use for retail analytics", "synonyms": ["store"]}}
    files = convert_osi_to_honeydew(_osi(model))
    ws = yaml.safe_load(files["workspace.yml"])
    assert any(s["name"] == "osi" for s in ws.get("metadata", []))


def test_osi_to_honeydew_relationship_on_from_entity_only():
    model = {"name": "m", "datasets": [
        {"name": "orders", "source": "db.s.orders", "fields": []},
        {"name": "customers", "source": "db.s.customers", "fields": []},
    ], "relationships": [{"name": "r", "from": "orders", "to": "customers",
        "from_columns": ["cid"], "to_columns": ["id"]}]}
    files = convert_osi_to_honeydew(_osi(model))
    orders = yaml.safe_load(files["schema/orders/orders.yml"])
    customers = yaml.safe_load(files["schema/customers/customers.yml"])
    assert len(orders["relations"]) == 1
    assert customers["relations"] == []
    rel = orders["relations"][0]
    assert rel["target_entity"] == "customers" and rel["rel_type"] == "many-to-one"
    assert rel["connection"] == [{"src_field": "cid", "target_field": "id"}]


def test_osi_to_honeydew_metric_assigned_by_expression_entity():
    model = {"name": "m",
        "datasets": [{"name": "orders", "source": "db.s.orders", "fields": []}],
        "metrics": [{"name": "total", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(orders.total)"}]}}]}
    files = convert_osi_to_honeydew(_osi(model))
    assert "schema/orders/metrics/total.yml" in files


def test_osi_to_honeydew_metric_entity_hint_overrides_expression():
    model = {"name": "m",
        "datasets": [
            {"name": "orders", "source": "db.s.orders", "fields": []},
            {"name": "customers", "source": "db.s.customers", "fields": []},
        ],
        "metrics": [{
            "name": "cnt",
            "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(orders.x)"}]},
            "custom_extensions": [{"vendor_name": "HONEYDEW", "data": '{"entity": "customers"}'}],
        }]}
    files = convert_osi_to_honeydew(_osi(model))
    assert "schema/customers/metrics/cnt.yml" in files
    assert "schema/orders/metrics/cnt.yml" not in files


def test_osi_to_honeydew_invalid_version_raises():
    with pytest.raises(HoneydewConversionError, match="Unsupported"):
        convert_osi_to_honeydew("version: '9.9.9'\nsemantic_model:\n  - name: m\n")


def test_osi_to_honeydew_missing_semantic_model_raises():
    with pytest.raises(HoneydewConversionError):
        convert_osi_to_honeydew(f"version: '{OSI_VERSION}'\n")


def test_osi_to_honeydew_subquery_source_uses_sql_type():
    model = {"name": "m", "datasets": [{"name": "orders",
        "source": "SELECT * FROM raw.orders WHERE active = true", "fields": []}]}
    files = convert_osi_to_honeydew(_osi(model))
    ds = yaml.safe_load(files["schema/orders/datasets/orders.yml"])
    assert ds["dataset_type"] == "sql"


def test_osi_to_honeydew_composite_primary_key():
    model = {"name": "m", "datasets": [{"name": "li", "source": "db.s.li",
        "primary_key": ["order_id", "line_number"], "fields": []}]}
    files = convert_osi_to_honeydew(_osi(model))
    entity = yaml.safe_load(files["schema/li/li.yml"])
    assert entity["keys"] == ["order_id", "line_number"]


def test_osi_to_honeydew_multiple_models_warns():
    doc = yaml.dump({"version": OSI_VERSION, "semantic_model": [
        {"name": "m1", "datasets": []},
        {"name": "m2", "datasets": []},
    ]})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        files = convert_osi_to_honeydew(doc)
    assert any("only the first" in str(x.message) for x in w)
    assert yaml.safe_load(files["workspace.yml"])["name"] == "m1"


# ─────────────────────────────────────────────────────────────────────────────
# Honeydew → OSI integration tests
# ─────────────────────────────────────────────────────────────────────────────

def test_honeydew_to_osi_basic(tmp_path):
    _write_workspace(str(tmp_path), "tpch", [{
        "name": "orders", "keys": ["orderkey"], "key_dataset": "tpch_orders",
        "sql": "SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS",
        "dataset_attrs": [
            {"column": "o_orderkey", "name": "orderkey", "datatype": "number"},
            {"column": "o_orderdate", "name": "orderdate", "datatype": "date"},
        ],
    }])
    result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    sm = result["semantic_model"][0]
    assert sm["name"] == "tpch"
    ds = sm["datasets"][0]
    assert ds["source"] == "SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS"
    assert ds["primary_key"] == ["orderkey"]


@pytest.mark.parametrize("col_name,datatype,expected_dim", [
    ("id", "number", None),
    ("status", "string", {"is_time": False}),
    ("created_at", "timestamp", {"is_time": True}),
])
def test_honeydew_to_osi_field_types(tmp_path, col_name, datatype, expected_dim):
    _write_workspace(str(tmp_path), "ws", [{"name": "orders", "keys": ["id"],
        "key_dataset": "orders", "sql": "db.s.orders",
        "dataset_attrs": [{"column": col_name, "name": col_name, "datatype": datatype}]}])
    result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    fields = {f["name"]: f for f in result["semantic_model"][0]["datasets"][0]["fields"]}
    assert fields[col_name].get("dimension") == expected_dim


def test_honeydew_to_osi_labels_become_label_and_ai_context(tmp_path):
    _write_workspace(str(tmp_path), "ws", [{"name": "orders", "keys": ["id"],
        "key_dataset": "orders", "sql": "db.s.orders",
        "dataset_attrs": [
            {"column": "status", "name": "status", "datatype": "string",
             "labels": ["sales", "reporting"]},
        ]}])
    result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    f = next(f for f in result["semantic_model"][0]["datasets"][0]["fields"] if f["name"] == "status")
    assert f["label"] == "sales"
    assert "sales" in (f.get("ai_context") or {}).get("synonyms", [])


def test_honeydew_to_osi_many_to_one_relation(tmp_path):
    _write_workspace(str(tmp_path), "ws", [
        {"name": "orders", "keys": ["order_id"], "key_dataset": "orders", "sql": "db.s.orders",
         "relations": [{"target_entity": "customers", "rel_type": "many-to-one",
                        "connection": [{"src_field": "customer_id", "target_field": "id"}]}],
         "dataset_attrs": []},
        {"name": "customers", "keys": ["id"], "key_dataset": "customers", "sql": "db.s.customers", "dataset_attrs": []},
    ])
    result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    rels = result["semantic_model"][0]["relationships"]
    assert len(rels) == 1
    assert rels[0]["from"] == "orders" and rels[0]["to"] == "customers"


def test_honeydew_to_osi_one_to_many_direction_flipped(tmp_path):
    _write_workspace(str(tmp_path), "ws", [
        {"name": "customers", "keys": ["id"], "key_dataset": "customers", "sql": "db.s.customers",
         "relations": [{"target_entity": "orders", "rel_type": "one-to-many",
                        "connection": [{"src_field": "id", "target_field": "customer_id"}]}],
         "dataset_attrs": []},
        {"name": "orders", "keys": ["order_id"], "key_dataset": "orders", "sql": "db.s.orders", "dataset_attrs": []},
    ])
    result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    rel = result["semantic_model"][0]["relationships"][0]
    assert rel["from"] == "orders" and rel["to"] == "customers"


def test_honeydew_to_osi_duplicate_relations_deduplicated(tmp_path):
    _write_workspace(str(tmp_path), "ws", [
        {"name": "orders", "keys": ["id"], "key_dataset": "orders", "sql": "db.s.orders",
         "relations": [{"target_entity": "customers", "rel_type": "many-to-one",
                        "connection": [{"src_field": "cid", "target_field": "id"}]}],
         "dataset_attrs": []},
        {"name": "customers", "keys": ["id"], "key_dataset": "customers", "sql": "db.s.customers",
         "relations": [{"target_entity": "orders", "rel_type": "one-to-many",
                        "connection": [{"src_field": "id", "target_field": "cid"}]}],
         "dataset_attrs": []},
    ])
    result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    assert len(result["semantic_model"][0].get("relationships", [])) == 1


def test_honeydew_to_osi_metric_converted(tmp_path):
    _write_workspace(str(tmp_path), "ws", [{"name": "orders", "keys": ["id"],
        "key_dataset": "orders", "sql": "db.s.orders", "dataset_attrs": [],
        "metrics": [{"type": "metric", "entity": "orders", "name": "count",
                     "datatype": "number", "sql": "COUNT(*)"}]}])
    result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    m = result["semantic_model"][0]["metrics"][0]
    assert m["name"] == "count"
    assert m["expression"]["dialects"][0]["expression"] == "COUNT(*)"


def test_honeydew_to_osi_metric_entity_preserved_in_extension(tmp_path):
    _write_workspace(str(tmp_path), "ws", [{"name": "orders", "keys": ["id"],
        "key_dataset": "orders", "sql": "db.s.orders", "dataset_attrs": [],
        "metrics": [{"type": "metric", "entity": "orders", "name": "cnt",
                     "datatype": "number", "sql": "COUNT(*)"}]}])
    result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    m = result["semantic_model"][0]["metrics"][0]
    ext = m["custom_extensions"][0]
    assert ext["vendor_name"] == "HONEYDEW"
    assert json.loads(ext["data"])["entity"] == "orders"


def test_honeydew_to_osi_calculated_attribute_as_field(tmp_path):
    _write_workspace(str(tmp_path), "ws", [{"name": "orders", "keys": ["id"],
        "key_dataset": "orders", "sql": "db.s.orders", "dataset_attrs": [],
        "calc_attrs": [{"type": "calculated_attribute", "entity": "orders",
                        "name": "discounted", "datatype": "number",
                        "sql": "orders.price * (1 - orders.discount)"}]}])
    result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    fields = {f["name"]: f for f in result["semantic_model"][0]["datasets"][0]["fields"]}
    assert "discounted" in fields
    assert "orders.price" in fields["discounted"]["expression"]["dialects"][0]["expression"]


def test_honeydew_to_osi_missing_workspace_raises(tmp_path):
    with pytest.raises(HoneydewConversionError, match="workspace.yml"):
        convert_honeydew_to_osi(str(tmp_path))


def test_honeydew_to_osi_missing_schema_dir_empty_model(tmp_path):
    (tmp_path / "workspace.yml").write_text(yaml.dump({"type": "workspace", "name": "ws"}))
    result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    assert result["semantic_model"][0]["datasets"] == []


def test_honeydew_to_osi_vendors_includes_honeydew(tmp_path):
    (tmp_path / "workspace.yml").write_text(yaml.dump({"type": "workspace", "name": "ws"}))
    result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    assert "HONEYDEW" in result.get("vendors", [])


def test_honeydew_to_osi_empty_metric_sql_skipped(tmp_path):
    _write_workspace(str(tmp_path), "ws", [{"name": "orders", "keys": ["id"],
        "key_dataset": "orders", "sql": "db.s.orders", "dataset_attrs": [],
        "metrics": [{"type": "metric", "entity": "orders", "name": "bad",
                     "datatype": "number", "sql": ""}]}])
    with warnings.catch_warnings(record=True):
        result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    assert "metrics" not in result["semantic_model"][0]


# ─────────────────────────────────────────────────────────────────────────────
# OSI → Honeydew → OSI round-trip tests
# ─────────────────────────────────────────────────────────────────────────────

def test_osi_roundtrip_name_and_description(tmp_path):
    model = {"name": "retail", "description": "Retail model", "datasets": []}
    sm = _osi_roundtrip(model, tmp_path)
    assert sm["name"] == "retail" and sm["description"] == "Retail model"


@pytest.mark.parametrize("primary_key", [
    ["order_id"],
    ["order_id", "line_no"],
])
def test_osi_roundtrip_primary_key(tmp_path, primary_key):
    model = {"name": "m", "datasets": [{"name": "orders", "source": "db.s.orders",
        "primary_key": primary_key, "fields": []}]}
    sm = _osi_roundtrip(model, tmp_path)
    assert sm["datasets"][0]["primary_key"] == primary_key


def test_osi_roundtrip_unique_keys(tmp_path):
    model = {"name": "m", "datasets": [{"name": "items", "source": "db.s.items",
        "primary_key": ["id"],
        "unique_keys": [["sku"], ["id", "variant"]],
        "fields": []}]}
    sm = _osi_roundtrip(model, tmp_path)
    assert sm["datasets"][0]["unique_keys"] == [["sku"], ["id", "variant"]]


def test_osi_roundtrip_field_label(tmp_path):
    model = {"name": "m", "datasets": [{"name": "orders", "source": "db.s.orders",
        "fields": [{"name": "status", "label": "sales",
                    "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "status"}]},
                    "dimension": {"is_time": False}}]}]}
    sm = _osi_roundtrip(model, tmp_path)
    f = next(f for f in sm["datasets"][0]["fields"] if f["name"] == "status")
    assert f["label"] == "sales"


def test_osi_roundtrip_ai_context_string(tmp_path):
    ai_ctx_value = "order status, order state"
    model = {"name": "m", "datasets": [{"name": "orders", "source": "db.s.orders",
        "fields": [{"name": "status",
                    "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "status"}]},
                    "ai_context": ai_ctx_value,
                    "dimension": {"is_time": False}}]}]}
    sm = _osi_roundtrip(model, tmp_path)
    f = next(f for f in sm["datasets"][0]["fields"] if f["name"] == "status")
    # String ai_context is merged into description on OSI→Honeydew; value must be recoverable
    assert ai_ctx_value in (f.get("description") or "") or f.get("ai_context") == ai_ctx_value


def test_osi_roundtrip_ai_context_dict(tmp_path):
    ctx = {"instructions": "Use for revenue analysis", "synonyms": ["revenue", "sales"]}
    model = {"name": "m", "datasets": [{"name": "orders", "source": "db.s.orders",
        "fields": [{"name": "total",
                    "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "total"}]},
                    "ai_context": ctx}]}]}
    sm = _osi_roundtrip(model, tmp_path)
    f = next(f for f in sm["datasets"][0]["fields"] if f["name"] == "total")
    assert f.get("ai_context") == ctx


def test_osi_roundtrip_model_ai_context(tmp_path):
    ctx = {"instructions": "Retail analytics", "synonyms": ["store"]}
    model = {"name": "m", "ai_context": ctx, "datasets": []}
    sm = _osi_roundtrip(model, tmp_path)
    assert sm.get("ai_context") == ctx


def test_osi_roundtrip_non_honeydew_custom_extensions(tmp_path):
    model = {"name": "m", "datasets": [{"name": "orders", "source": "db.s.orders",
        "custom_extensions": [{"vendor_name": "SNOWFLAKE", "data": '{"warehouse": "WH"}'}],
        "fields": []}]}
    sm = _osi_roundtrip(model, tmp_path)
    exts = sm["datasets"][0].get("custom_extensions") or []
    assert any(e["vendor_name"] == "SNOWFLAKE" for e in exts)


def test_osi_roundtrip_relationship_name(tmp_path):
    model = {"name": "m", "datasets": [
        {"name": "orders", "source": "db.s.orders", "fields": []},
        {"name": "customers", "source": "db.s.customers", "fields": []},
    ], "relationships": [{"name": "orders_to_customers", "from": "orders", "to": "customers",
        "from_columns": ["cid"], "to_columns": ["id"]}]}
    sm = _osi_roundtrip(model, tmp_path)
    assert sm["relationships"][0]["name"] == "orders_to_customers"


def test_osi_roundtrip_relationship_columns(tmp_path):
    model = {"name": "m", "datasets": [
        {"name": "orders", "source": "db.s.orders", "fields": []},
        {"name": "customers", "source": "db.s.customers", "fields": []},
    ], "relationships": [{"name": "r", "from": "orders", "to": "customers",
        "from_columns": ["cid"], "to_columns": ["id"]}]}
    sm = _osi_roundtrip(model, tmp_path)
    rel = sm["relationships"][0]
    assert rel["from_columns"] == ["cid"] and rel["to_columns"] == ["id"]


def test_osi_roundtrip_metric(tmp_path):
    model = {"name": "m",
        "datasets": [{"name": "orders", "source": "db.s.orders", "fields": []}],
        "metrics": [{"name": "total_revenue", "description": "Sum of sales",
                     "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(orders.total)"}]}}]}
    sm = _osi_roundtrip(model, tmp_path)
    m = sm["metrics"][0]
    assert m["name"] == "total_revenue"
    assert m["expression"]["dialects"][0]["expression"] == "SUM(orders.total)"
    assert m["description"] == "Sum of sales"


def test_osi_roundtrip_tpcds_example(tmp_path):
    tpcds_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "examples" / "tpcds_semantic_model.yaml"
    )
    if not tpcds_path.exists():
        pytest.skip("TPC-DS example not found")
    osi_yaml = tpcds_path.read_text()
    files = convert_osi_to_honeydew(osi_yaml)
    for rel_path, content in files.items():
        p = tmp_path / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    sm = result["semantic_model"][0]
    assert sm["name"] == "tpcds_retail_model"
    ds_names = {ds["name"] for ds in sm["datasets"]}
    assert "store_sales" in ds_names and "customer" in ds_names


# ─────────────────────────────────────────────────────────────────────────────
# Honeydew → OSI → Honeydew round-trip tests
# ─────────────────────────────────────────────────────────────────────────────

def test_honeydew_roundtrip_entity_name_and_keys(tmp_path):
    out_dir = _honeydew_roundtrip([{
        "name": "orders", "keys": ["order_id"], "key_dataset": "orders",
        "sql": "DB.S.ORDERS", "dataset_attrs": [],
    }], tmp_path)
    entity = yaml.safe_load((out_dir / "schema/orders/orders.yml").read_text())
    assert entity["name"] == "orders" and entity["keys"] == ["order_id"]


def test_honeydew_roundtrip_source(tmp_path):
    out_dir = _honeydew_roundtrip([{
        "name": "orders", "keys": ["id"], "key_dataset": "orders",
        "sql": "DB.SCHEMA.ORDERS", "dataset_attrs": [],
    }], tmp_path)
    ds = yaml.safe_load((out_dir / "schema/orders/datasets/orders.yml").read_text())
    assert ds["sql"] == "DB.SCHEMA.ORDERS"


def test_honeydew_roundtrip_column_attributes(tmp_path):
    out_dir = _honeydew_roundtrip([{
        "name": "orders", "keys": ["id"], "key_dataset": "orders",
        "sql": "DB.S.ORDERS",
        "dataset_attrs": [
            {"column": "o_id", "name": "id", "datatype": "number"},
            {"column": "o_status", "name": "status", "datatype": "string"},
        ],
    }], tmp_path)
    ds = yaml.safe_load((out_dir / "schema/orders/datasets/orders.yml").read_text())
    attrs = {a["name"]: a for a in ds["attributes"]}
    assert attrs["id"]["column"] == "o_id"
    assert attrs["status"]["datatype"] == "string"


def test_honeydew_roundtrip_labels_on_column(tmp_path):
    out_dir = _honeydew_roundtrip([{
        "name": "orders", "keys": ["id"], "key_dataset": "orders",
        "sql": "DB.S.ORDERS",
        "dataset_attrs": [{"column": "status", "name": "status", "datatype": "string", "labels": ["sales"]}],
    }], tmp_path)
    ds = yaml.safe_load((out_dir / "schema/orders/datasets/orders.yml").read_text())
    attrs = {a["name"]: a for a in ds["attributes"]}
    assert "sales" in attrs["status"].get("labels", [])


def test_honeydew_roundtrip_calculated_attribute_sql(tmp_path):
    out_dir = _honeydew_roundtrip([{
        "name": "orders", "keys": ["id"], "key_dataset": "orders",
        "sql": "DB.S.ORDERS", "dataset_attrs": [],
        "calc_attrs": [{"type": "calculated_attribute", "entity": "orders",
                        "name": "disc", "datatype": "number",
                        "sql": "orders.price * (1 - orders.discount)"}],
    }], tmp_path)
    calc = yaml.safe_load((out_dir / "schema/orders/attributes/disc.yml").read_text())
    assert calc["sql"] == "orders.price * (1 - orders.discount)"


def test_honeydew_roundtrip_metric_entity_assignment(tmp_path):
    out_dir = _honeydew_roundtrip([{
        "name": "orders", "keys": ["id"], "key_dataset": "orders",
        "sql": "DB.S.ORDERS", "dataset_attrs": [],
        "metrics": [{"type": "metric", "entity": "orders", "name": "cnt",
                     "datatype": "number", "sql": "COUNT(*)"}],
    }], tmp_path)
    m = yaml.safe_load((out_dir / "schema/orders/metrics/cnt.yml").read_text())
    assert m["entity"] == "orders" and m["sql"] == "COUNT(*)"


def test_honeydew_roundtrip_relation(tmp_path):
    out_dir = _honeydew_roundtrip([
        {"name": "orders", "keys": ["id"], "key_dataset": "orders", "sql": "DB.S.ORDERS",
         "relations": [{"target_entity": "customers", "rel_type": "many-to-one",
                        "connection": [{"src_field": "cid", "target_field": "id"}]}],
         "dataset_attrs": []},
        {"name": "customers", "keys": ["id"], "key_dataset": "customers",
         "sql": "DB.S.CUSTOMERS", "dataset_attrs": []},
    ], tmp_path)
    entity = yaml.safe_load((out_dir / "schema/orders/orders.yml").read_text())
    assert entity["relations"][0]["target_entity"] == "customers"
    assert entity["relations"][0]["connection"][0]["src_field"] == "cid"


def test_honeydew_roundtrip_bool_datatype(tmp_path):
    out_dir = _honeydew_roundtrip([{
        "name": "orders", "keys": ["id"], "key_dataset": "orders",
        "sql": "DB.S.ORDERS",
        "dataset_attrs": [{"column": "is_active", "name": "is_active", "datatype": "bool"}],
    }], tmp_path)
    ds = yaml.safe_load((out_dir / "schema/orders/datasets/orders.yml").read_text())
    attrs = {a["name"]: a for a in ds["attributes"]}
    assert attrs["is_active"]["datatype"] == "bool"


def test_honeydew_roundtrip_connection_expr(tmp_path):
    out_dir = _honeydew_roundtrip([
        {"name": "orders", "keys": ["id"], "key_dataset": "orders", "sql": "DB.S.ORDERS",
         "relations": [{"target_entity": "customers", "rel_type": "many-to-one",
                        "connection_expr": {"sql": "orders.cid = customers.id AND orders.region = customers.region"}}],
         "dataset_attrs": []},
        {"name": "customers", "keys": ["id"], "key_dataset": "customers",
         "sql": "DB.S.CUSTOMERS", "dataset_attrs": []},
    ], tmp_path)
    entity = yaml.safe_load((out_dir / "schema/orders/orders.yml").read_text())
    rel = entity["relations"][0]
    assert rel.get("connection_expr", {}).get("sql") == "orders.cid = customers.id AND orders.region = customers.region"


@pytest.mark.parametrize("attr_extra,check_key,check_val", [
    ({"display_name": "Order Status"}, "display_name", "Order Status"),
    ({"hidden": True}, "hidden", True),
    ({"format_string": "##,###"}, "format_string", "##,###"),
])
def test_honeydew_roundtrip_dataset_attr_honeydew_field(tmp_path, attr_extra, check_key, check_val):
    attr = {"column": "status", "name": "status", "datatype": "string", **attr_extra}
    out_dir = _honeydew_roundtrip([{
        "name": "orders", "keys": ["id"], "key_dataset": "orders",
        "sql": "DB.S.ORDERS", "dataset_attrs": [attr],
    }], tmp_path)
    ds = yaml.safe_load((out_dir / "schema/orders/datasets/orders.yml").read_text())
    attrs = {a["name"]: a for a in ds["attributes"]}
    assert attrs["status"][check_key] == check_val


@pytest.mark.parametrize("calc_extra,check_key,check_val", [
    ({"display_name": "Discounted Price"}, "display_name", "Discounted Price"),
    ({"timegrain": "day"}, "timegrain", "day"),
])
def test_honeydew_roundtrip_calc_attr_honeydew_field(tmp_path, calc_extra, check_key, check_val):
    calc = {"type": "calculated_attribute", "entity": "orders",
            "name": "disc", "datatype": "number", "sql": "orders.price * 0.9", **calc_extra}
    out_dir = _honeydew_roundtrip([{
        "name": "orders", "keys": ["id"], "key_dataset": "orders",
        "sql": "DB.S.ORDERS", "dataset_attrs": [], "calc_attrs": [calc],
    }], tmp_path)
    result = yaml.safe_load((out_dir / "schema/orders/attributes/disc.yml").read_text())
    assert result[check_key] == check_val


@pytest.mark.parametrize("entity_extra,check_key,check_val", [
    ({"owner": "analytics_team"}, "owner", "analytics_team"),
    ({"display_name": "Orders Table"}, "display_name", "Orders Table"),
    ({"hidden": True}, "hidden", True),
    ({"folder": "finance"}, "folder", "finance"),
])
def test_honeydew_roundtrip_entity_honeydew_field(tmp_path, entity_extra, check_key, check_val):
    out_dir = _honeydew_roundtrip([{
        "name": "orders", "keys": ["id"], "key_dataset": "orders",
        "sql": "DB.S.ORDERS", "dataset_attrs": [], **entity_extra,
    }], tmp_path)
    entity = yaml.safe_load((out_dir / "schema/orders/orders.yml").read_text())
    assert entity.get(check_key) == check_val


def test_honeydew_roundtrip_calc_attr_simple_identifier_stays_calc(tmp_path):
    out_dir = _honeydew_roundtrip([{
        "name": "orders", "keys": ["id"], "key_dataset": "orders",
        "sql": "DB.S.ORDERS", "dataset_attrs": [],
        "calc_attrs": [{"type": "calculated_attribute", "entity": "orders",
                        "name": "revenue", "datatype": "number", "sql": "revenue"}],
    }], tmp_path)
    calc_path = out_dir / "schema/orders/attributes/revenue.yml"
    assert calc_path.exists(), "calculated_attribute with simple-id sql should not become a dataset column"
    calc = yaml.safe_load(calc_path.read_text())
    assert calc["sql"] == "revenue"


# ─────────────────────────────────────────────────────────────────────────────
# Bug-fix regression tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("expression", [
    {"dialects": [{"dialect": "ANSI_SQL", "expression": ""}]},
    {"dialects": [{"dialect": "ANSI_SQL", "expression": "   "}]},
])
def test_empty_or_whitespace_field_expression_skipped(expression):
    model = {"name": "m", "datasets": [{"name": "orders", "source": "db.s.orders", "fields": [{
        "name": "bad",
        "expression": expression,
        "dimension": {"is_time": False},
    }]}]}
    files = convert_osi_to_honeydew(_osi(model))
    ds = yaml.safe_load(files["schema/orders/datasets/orders.yml"])
    assert all(a["name"] != "bad" for a in ds["attributes"])
    assert "schema/orders/attributes/bad.yml" not in files


@pytest.mark.parametrize("expression", [
    {"dialects": [{"dialect": "ANSI_SQL", "expression": ""}]},
    {"dialects": [{"dialect": "ANSI_SQL", "expression": "   "}]},
])
def test_empty_or_whitespace_metric_expression_skipped(expression):
    model = {"name": "m",
        "datasets": [{"name": "orders", "source": "db.s.orders", "fields": []}],
        "metrics": [{"name": "bad_m", "expression": expression}]}
    files = convert_osi_to_honeydew(_osi(model))
    assert "schema/orders/metrics/bad_m.yml" not in files


def test_non_dict_expression_warns():
    model = {"name": "m", "datasets": [{"name": "orders", "source": "db.s.orders", "fields": [{
        "name": "bad",
        "expression": "just_a_string",
        "dimension": {"is_time": False},
    }]}]}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        files = convert_osi_to_honeydew(_osi(model))
    assert any("must be a mapping" in str(x.message) for x in w)
    ds = yaml.safe_load(files["schema/orders/datasets/orders.yml"])
    assert all(a["name"] != "bad" for a in ds["attributes"])


def test_duplicate_metric_name_warns():
    model = {"name": "m",
        "datasets": [{"name": "orders", "source": "db.s.orders", "fields": []}],
        "metrics": [
            {"name": "total", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(orders.a)"}]}},
            {"name": "total", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(orders.b)"}]}},
        ]}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        files = convert_osi_to_honeydew(_osi(model))
    assert any("total" in str(x.message) for x in w)
    m = yaml.safe_load(files["schema/orders/metrics/total.yml"])
    assert "orders.b" in m["sql"]


def test_metric_string_ai_context_preserved_in_roundtrip(tmp_path):
    model = {"name": "m",
        "datasets": [{"name": "orders", "source": "db.s.orders", "fields": []}],
        "metrics": [{"name": "rev", "ai_context": "Use for revenue analysis",
                     "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(orders.total)"}]}}]}
    files = convert_osi_to_honeydew(_osi(model))
    for rel_path, content in files.items():
        p = tmp_path / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    m = result["semantic_model"][0]["metrics"][0]
    assert m.get("ai_context") == "Use for revenue analysis"


def test_malformed_osi_metadata_json_warns(tmp_path):
    ws_path = tmp_path / "workspace.yml"
    ws_path.write_text(yaml.dump({"type": "workspace", "name": "ws"}))
    base = tmp_path / "schema" / "orders"
    (base / "datasets").mkdir(parents=True)
    entity = {
        "type": "entity", "name": "orders", "keys": ["id"], "key_dataset": "orders",
        "relations": [],
        "metadata": [{"name": "osi", "metadata": [
            {"name": "unique_keys", "value": "[broken json"},
        ]}],
    }
    (base / "orders.yml").write_text(yaml.dump(entity))
    (base / "datasets" / "orders.yml").write_text(yaml.dump(
        {"type": "dataset", "entity": "orders", "name": "orders",
         "sql": "DB.S.ORDERS", "dataset_type": "table", "attributes": []}
    ))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        convert_honeydew_to_osi(str(tmp_path))
    assert any("unique_keys" in str(x.message) for x in w)


# ─────────────────────────────────────────────────────────────────────────────
# _fields_to_honeydew unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_fields_to_honeydew_simple_identifier_goes_to_dataset():
    fields = [{"name": "status", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "status"}]},
               "dimension": {"is_time": False}}]
    dataset_attrs, calc_attrs = _fields_to_honeydew(fields, "orders")
    assert len(dataset_attrs) == 1 and len(calc_attrs) == 0
    assert dataset_attrs[0]["column"] == "status"


def test_fields_to_honeydew_complex_sql_goes_to_calc():
    fields = [{"name": "disc", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "price * 0.9"}]}}]
    dataset_attrs, calc_attrs = _fields_to_honeydew(fields, "orders")
    assert len(dataset_attrs) == 0 and len(calc_attrs) == 1
    assert calc_attrs[0]["sql"] == "price * 0.9"


def test_fields_to_honeydew_missing_name_raises():
    fields = [{"expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "col"}]}}]
    with pytest.raises(HoneydewConversionError, match="missing 'name'"):
        _fields_to_honeydew(fields, "orders")


def test_fields_to_honeydew_empty_expression_skipped():
    fields = [{"name": "bad", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": ""}]}}]
    dataset_attrs, calc_attrs = _fields_to_honeydew(fields, "orders")
    assert dataset_attrs == [] and calc_attrs == []


# ─────────────────────────────────────────────────────────────────────────────
# Connectionless relation warning
# ─────────────────────────────────────────────────────────────────────────────

def test_connectionless_relation_warns():
    model = {"name": "m", "datasets": [
        {"name": "orders", "source": "db.s.orders", "fields": []},
        {"name": "customers", "source": "db.s.customers", "fields": []},
    ], "relationships": [{"name": "r", "from": "orders", "to": "customers"}]}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        files = convert_osi_to_honeydew(_osi(model))
    assert any("resolve the join" in str(x.message) for x in w)
    entity = yaml.safe_load(files["schema/orders/orders.yml"])
    assert entity["relations"][0]["target_entity"] == "customers"


# ─────────────────────────────────────────────────────────────────────────────
# vendors round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_vendors_roundtrip_preserves_non_honeydew(tmp_path):
    doc = yaml.dump({
        "version": OSI_VERSION,
        "vendors": ["SNOWFLAKE", "HONEYDEW"],
        "semantic_model": [{"name": "m", "datasets": []}],
    })
    files = convert_osi_to_honeydew(doc)
    for rel_path, content in files.items():
        p = tmp_path / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    assert "SNOWFLAKE" in result["vendors"]
    assert "HONEYDEW" in result["vendors"]


def test_vendors_always_includes_honeydew(tmp_path):
    doc = yaml.dump({
        "version": OSI_VERSION,
        "vendors": ["SNOWFLAKE"],
        "semantic_model": [{"name": "m", "datasets": []}],
    })
    files = convert_osi_to_honeydew(doc)
    for rel_path, content in files.items():
        p = tmp_path / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    result = yaml.safe_load(convert_honeydew_to_osi(str(tmp_path)))
    assert result["vendors"][0] == "HONEYDEW"


# ─────────────────────────────────────────────────────────────────────────────
# main() CLI smoke tests
# ─────────────────────────────────────────────────────────────────────────────

def test_main_osi_to_honeydew(tmp_path):
    import subprocess, sys
    input_file = tmp_path / "model.yaml"
    input_file.write_text(yaml.dump({
        "version": OSI_VERSION,
        "semantic_model": [{"name": "m", "datasets": [
            {"name": "orders", "source": "db.s.orders", "fields": []}
        ]}],
    }))
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent.parent / "src" / "honeydew_osi_converter.py"),
         "osi-to-honeydew", "-i", str(input_file), "-o", str(output_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert (output_dir / "workspace.yml").exists()
    ws = yaml.safe_load((output_dir / "workspace.yml").read_text())
    assert ws["name"] == "m"


def test_main_honeydew_to_osi(tmp_path):
    import subprocess, sys
    _write_workspace(str(tmp_path), "ws", [{
        "name": "orders", "keys": ["id"], "key_dataset": "orders",
        "sql": "DB.S.ORDERS", "dataset_attrs": [],
    }])
    output_file = tmp_path / "output.yaml"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent.parent / "src" / "honeydew_osi_converter.py"),
         "honeydew-to-osi", "-i", str(tmp_path), "-o", str(output_file)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert output_file.exists()
    doc = yaml.safe_load(output_file.read_text())
    assert doc["semantic_model"][0]["name"] == "ws"


def test_main_path_traversal_rejected(tmp_path):
    import subprocess, sys
    # Entity name containing traversal sequences generates paths that escape output_dir
    input_file = tmp_path / "model.yaml"
    input_file.write_text(
        f"version: '{OSI_VERSION}'\nsemantic_model:\n"
        "  - name: m\n    datasets:\n"
        "      - name: '../../evil'\n        source: db.s.evil\n        fields: []\n"
    )
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent.parent / "src" / "honeydew_osi_converter.py"),
         "osi-to-honeydew", "-i", str(input_file), "-o", str(output_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "refusing to write" in result.stderr
    assert not (tmp_path / "evil.yml").exists()
