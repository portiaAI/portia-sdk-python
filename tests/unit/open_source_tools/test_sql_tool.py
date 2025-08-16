"""Unit tests for the SQLTool using the default SQLite adapter."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from portia.errors import ToolSoftError
from portia.open_source_tools.sql_tool import SQLTool
from tests.utils import get_test_tool_context

if TYPE_CHECKING:  
    from portia.tool import ToolRunContext


def make_ctx() -> ToolRunContext:
    """Construct a ToolRunContext using the shared test helper."""
    return get_test_tool_context()


def create_temp_sqlite_with_schema() -> Path:
    """Create a temporary SQLite DB with a simple users table and data."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        tmp_path = Path(tmp.name)

    conn = sqlite3.connect(tmp_path.as_posix())
    try:
        cur = conn.cursor()
        cur.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        cur.executemany(
            "INSERT INTO users (name, age) VALUES (?, ?)",
            [("alice", 30), ("bob", 25), ("carol", 35)],
        )
        conn.commit()
    finally:
        conn.close()
    return tmp_path


def test_run_sql_select_rows() -> None:
    """run_sql returns row dicts for a valid SELECT query."""
    db_path = create_temp_sqlite_with_schema()
    try:
        tool = SQLTool()
        ctx = make_ctx()
        result_output = tool._run(
            ctx,
            action="run_sql",
            query="SELECT name, age FROM users WHERE age > 28 ORDER BY age",
            config_json=json.dumps({"db_path": db_path.as_posix()}),
        )
        rows = result_output.get_value()
        assert isinstance(rows, list)
        assert rows[0]["name"] == "alice"
        assert rows[-1]["name"] == "carol"
    finally:
        db_path.unlink()


def test_run_sql_reject_non_select() -> None:
    """Non-SELECT queries are rejected with ToolSoftError."""
    db_path = create_temp_sqlite_with_schema()
    try:
        tool = SQLTool()
        ctx = make_ctx()
        with pytest.raises(ToolSoftError):
            tool._run(
                ctx,
                action="run_sql",
                query="UPDATE users SET age = 99",
                config_json=json.dumps({"db_path": db_path.as_posix()}),
            )
    finally:
        db_path.unlink()


def test_list_tables() -> None:
    """list_tables returns created tables."""
    db_path = create_temp_sqlite_with_schema()
    try:
        tool = SQLTool()
        ctx = make_ctx()
        result_output = tool._run(
            ctx, action="list_tables", config_json=json.dumps({"db_path": db_path.as_posix()})
        )
        tables = result_output.get_value()
        assert "users" in tables
    finally:
        db_path.unlink()


def test_get_table_schemas() -> None:
    """get_table_schemas returns PRAGMA table_info for requested tables."""
    db_path = create_temp_sqlite_with_schema()
    try:
        tool = SQLTool()
        ctx = make_ctx()
        result_output = tool._run(
            ctx,
            action="get_table_schemas",
            tables=["users"],
            config_json=json.dumps({"db_path": db_path.as_posix()}),
        )
        schemas = result_output.get_value()
        assert "users" in schemas
        cols = schemas["users"]
        names = [c["name"] for c in cols]
        assert {"id", "name", "age"}.issubset(set(names))
    finally:
        db_path.unlink()


def test_check_sql_valid_and_invalid() -> None:
    """check_sql returns ok True for valid SQL and False with error for invalid."""
    db_path = create_temp_sqlite_with_schema()
    try:
        tool = SQLTool()
        ctx = make_ctx()
        ok = tool._run(
            ctx,
            action="check_sql",
            query="SELECT * FROM users WHERE age >= 30",
            config_json=json.dumps({"db_path": db_path.as_posix()}),
        ).get_value()
        assert ok["ok"] is True

        bad = tool._run(
            ctx,
            action="check_sql",
            query="SELECT * FROM not_a_table",
            config_json=json.dumps({"db_path": db_path.as_posix()}),
        ).get_value()
        assert bad["ok"] is False
        assert "error" in bad
    finally:
        db_path.unlink()
