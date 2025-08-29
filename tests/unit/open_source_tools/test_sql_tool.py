"""Unit tests for the SQL tools using the default SQLite adapter."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from portia.open_source_tools.sql_tool import (
    RunSQLTool,
    ListTablesTool,
    GetTableSchemasTool,
    CheckSQLTool,
)
from tests.utils import get_test_tool_context

if TYPE_CHECKING:  
    from portia.tool import ToolRunContext


@pytest.fixture
def test_context() -> ToolRunContext:
    """Construct a ToolRunContext using the shared test helper."""
    return get_test_tool_context()


@pytest.fixture
def temp_sqlite_db() -> Path:
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
    
    yield tmp_path
    
    # Cleanup
    tmp_path.unlink(missing_ok=True)




# Tests for the new specific tools

def test_run_sql_tool(temp_sqlite_db: Path, test_context: ToolRunContext) -> None:
    """RunSQLTool executes SELECT queries and returns results."""
    tool = RunSQLTool()
    result_output = tool._run(
        test_context,
        query="SELECT name, age FROM users WHERE age > 28 ORDER BY age",
        config_json=json.dumps({"db_path": temp_sqlite_db.as_posix()}),
    )
    rows = result_output.get_value()
    assert isinstance(rows, list)
    assert len(rows) == 2
    assert rows[0]["name"] == "alice"
    assert rows[0]["age"] == 30
    assert rows[1]["name"] == "carol"
    assert rows[1]["age"] == 35


def test_run_sql_tool_authorization_error(temp_sqlite_db: Path, test_context: ToolRunContext) -> None:
    """RunSQLTool rejects non-SELECT queries via SQLite authorizer."""
    tool = RunSQLTool()
    # The authorizer should prevent UPDATE operations
    with pytest.raises(Exception):  # SQLite will raise authorization error
        tool._run(
            test_context,
            query="UPDATE users SET age = 99",
            config_json=json.dumps({"db_path": temp_sqlite_db.as_posix()}),
        )


def test_list_tables_tool(temp_sqlite_db: Path, test_context: ToolRunContext) -> None:
    """ListTablesTool returns all available tables."""
    tool = ListTablesTool()
    result_output = tool._run(
        test_context,
        config_json=json.dumps({"db_path": temp_sqlite_db.as_posix()}),
    )
    tables = result_output.get_value()
    assert isinstance(tables, list)
    assert "users" in tables


def test_get_table_schemas_tool(temp_sqlite_db: Path, test_context: ToolRunContext) -> None:
    """GetTableSchemasTool returns complete schema information for requested tables."""
    tool = GetTableSchemasTool()
    result_output = tool._run(
        test_context,
        tables=["users"],
        config_json=json.dumps({"db_path": temp_sqlite_db.as_posix()}),
    )
    schemas = result_output.get_value()
    assert "users" in schemas
    cols = schemas["users"]
    
    # Test complete expected output structure
    expected_columns = [
        {"cid": 0, "name": "id", "type": "INTEGER", "notnull": 0, "dflt_value": None, "pk": 1},
        {"cid": 1, "name": "name", "type": "TEXT", "notnull": 0, "dflt_value": None, "pk": 0},
        {"cid": 2, "name": "age", "type": "INTEGER", "notnull": 0, "dflt_value": None, "pk": 0},
    ]
    assert cols == expected_columns


def test_check_sql_tool_valid_and_invalid(temp_sqlite_db: Path, test_context: ToolRunContext) -> None:
    """CheckSQLTool validates queries without executing them."""
    tool = CheckSQLTool()
    
    # Valid query
    ok_result = tool._run(
        test_context,
        query="SELECT * FROM users WHERE age >= 30",
        config_json=json.dumps({"db_path": temp_sqlite_db.as_posix()}),
    ).get_value()
    assert ok_result == {"ok": True}

    # Invalid query  
    bad_result = tool._run(
        test_context,
        query="SELECT * FROM not_a_table",
        config_json=json.dumps({"db_path": temp_sqlite_db.as_posix()}),
    ).get_value()
    assert bad_result["ok"] is False
    assert "error" in bad_result
    assert "no such table: not_a_table" in bad_result["error"]
