"""Unit tests for the SQL tools using the default SQLite adapter."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from portia.errors import ToolHardError, ToolSoftError
from portia.open_source_tools.sql_tool import (
    RunSQLTool,
    ListTablesTool,
    GetTableSchemasTool,
    CheckSQLTool,
    SQLAdapter,
    SQLiteAdapter,
    SQLiteConfig,
    _sqlite_authorizer,
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


def test_sqlite_authorizer() -> None:
    """Test the SQLite authorizer function covers all branches."""
    # Test allowed operations
    assert _sqlite_authorizer(sqlite3.SQLITE_READ, None, None, None, None) == sqlite3.SQLITE_OK
    assert _sqlite_authorizer(sqlite3.SQLITE_SELECT, None, None, None, None) == sqlite3.SQLITE_OK
    assert _sqlite_authorizer(sqlite3.SQLITE_FUNCTION, None, None, None, None) == sqlite3.SQLITE_OK
    
    # Test allowed PRAGMA operations
    assert _sqlite_authorizer(sqlite3.SQLITE_PRAGMA, "table_info", None, None, None) == sqlite3.SQLITE_OK
    assert _sqlite_authorizer(sqlite3.SQLITE_PRAGMA, "table_list", None, None, None) == sqlite3.SQLITE_OK
    assert _sqlite_authorizer(sqlite3.SQLITE_PRAGMA, "database_list", None, None, None) == sqlite3.SQLITE_OK
    
    # Test denied PRAGMA operations (covers line 70)
    assert _sqlite_authorizer(sqlite3.SQLITE_PRAGMA, "journal_mode", None, None, None) == sqlite3.SQLITE_DENY
    assert _sqlite_authorizer(sqlite3.SQLITE_PRAGMA, "foreign_keys", None, None, None) == sqlite3.SQLITE_DENY
    
    # Test denied operations
    assert _sqlite_authorizer(sqlite3.SQLITE_INSERT, None, None, None, None) == sqlite3.SQLITE_DENY
    assert _sqlite_authorizer(sqlite3.SQLITE_UPDATE, None, None, None, None) == sqlite3.SQLITE_DENY
    assert _sqlite_authorizer(sqlite3.SQLITE_DELETE, None, None, None, None) == sqlite3.SQLITE_DENY


def test_sql_adapter_abstract_methods() -> None:
    """Test that abstract SQLAdapter methods raise NotImplementedError."""
    adapter = SQLAdapter()
    
    with pytest.raises(NotImplementedError):
        adapter.run_sql("SELECT 1")
    
    with pytest.raises(NotImplementedError):
        adapter.list_tables()
    
    with pytest.raises(NotImplementedError):
        adapter.get_table_schemas(["test"])
    
    with pytest.raises(NotImplementedError):
        adapter.check_sql("SELECT 1")


def test_sqlite_adapter_memory_db(test_context: ToolRunContext) -> None:
    """Test SQLiteAdapter with in-memory database."""
    # Create a temporary file-based database for setup since in-memory with authorizer 
    # prevents CREATE operations
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        # Set up test data using a normal connection without authorizer
        setup_conn = sqlite3.connect(tmp_path.as_posix())
        setup_conn.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
        setup_conn.execute("INSERT INTO test_table (id, name) VALUES (1, 'test')")
        setup_conn.commit()
        setup_conn.close()
        
        # Now test with the adapter
        config = SQLiteConfig(db_path=tmp_path.as_posix())
        adapter = SQLiteAdapter(config)
        
        # Test run_sql
        result = adapter.run_sql("SELECT * FROM test_table")
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["name"] == "test"
        
        # Test list_tables
        tables = adapter.list_tables()
        assert "test_table" in tables
        
        # Test get_table_schemas
        schemas = adapter.get_table_schemas(["test_table"])
        assert "test_table" in schemas
        assert len(schemas["test_table"]) == 2  # id and name columns
        
        # Test check_sql
        result = adapter.check_sql("SELECT * FROM test_table")
        assert result["ok"] is True
        
        # Test the memory database connection path 
        # by creating an in-memory adapter and testing basic operations
        memory_config = SQLiteConfig(db_path=":memory:")
        memory_adapter = SQLiteAdapter(memory_config)
        
        # This will test the :memory: path in _connect() but won't work for CREATE
        # since the authorizer prevents it. We can test connection establishment instead.
        with memory_adapter._connect() as conn:
            # Test that we can connect and the connection works
            cursor = conn.execute("SELECT 1 as test_value")
            result = cursor.fetchone()
            assert result[0] == 1
            
    finally:
        # Cleanup
        tmp_path.unlink(missing_ok=True)



def test_sqlite_adapter_error_handling() -> None:
    """Test SQLiteAdapter error handling for database operations."""
    # Use non-existent database path to trigger SQLite errors
    config = SQLiteConfig(db_path="/non/existent/path/database.db")
    adapter = SQLiteAdapter(config)
    
    # Test run_sql error handling
    with pytest.raises(ToolHardError) as exc_info:
        adapter.run_sql("SELECT 1")
    assert "SQLite error" in str(exc_info.value)
    
    # Test list_tables error handling 
    with pytest.raises(ToolHardError) as exc_info:
        adapter.list_tables()
    assert "SQLite error" in str(exc_info.value)
    
    # Test get_table_schemas error handling 
    with pytest.raises(ToolHardError) as exc_info:
        adapter.get_table_schemas(["test_table"])
    assert "SQLite error" in str(exc_info.value)


# Test environment variable configuration 
def test_adapter_from_env() -> None:
    """Test adapter creation from environment variables."""
    tool = RunSQLTool()
    
    # Test with SQLITE_DB_PATH set
    with mock.patch.dict(os.environ, {"SQLITE_DB_PATH": "/tmp/test.db"}):
        adapter = tool._adapter_from_env()
        assert isinstance(adapter, SQLiteAdapter)
        assert adapter.config.db_path == "/tmp/test.db"
    
    # Test without SQLITE_DB_PATH (- default to :memory:)
    with mock.patch.dict(os.environ, {}, clear=True):
        adapter = tool._adapter_from_env()
        assert isinstance(adapter, SQLiteAdapter)
        assert adapter.config.db_path == ":memory:"


# Test config_json parsing 
def test_adapter_from_config_json() -> None:
    """Test adapter creation from config_json parameter."""
    tool = RunSQLTool()
    
    # Test with None config_json 
    adapter = tool._adapter_from_config_json(None)
    assert adapter is None
    
    # Test with empty config_json 
    adapter = tool._adapter_from_config_json("")
    assert adapter is None
    
    # Test with invalid JSON 
    with pytest.raises(ToolSoftError) as exc_info:
        tool._adapter_from_config_json("invalid json")
    assert "Invalid config_json" in str(exc_info.value)
    
    # Test with valid JSON but missing db_path
    with pytest.raises(ToolSoftError) as exc_info:
        tool._adapter_from_config_json(json.dumps({"other_key": "value"}))
    assert "config_json must include a non-empty 'db_path'" in str(exc_info.value)
    
    # Test with empty db_path 
    with pytest.raises(ToolSoftError) as exc_info:
        tool._adapter_from_config_json(json.dumps({"db_path": ""}))
    assert "config_json must include a non-empty 'db_path'" in str(exc_info.value)
    
    # Test with non-string db_path 
    with pytest.raises(ToolSoftError) as exc_info:
        tool._adapter_from_config_json(json.dumps({"db_path": 123}))
    assert "config_json must include a non-empty 'db_path'" in str(exc_info.value)
    
    # Test with valid config_json
    adapter = tool._adapter_from_config_json(json.dumps({"db_path": "/tmp/test.db"}))
    assert isinstance(adapter, SQLiteAdapter)
    assert adapter.config.db_path == "/tmp/test.db"


# Test tools using environment variables (no config_json)
def test_tools_with_env_config(test_context: ToolRunContext, temp_sqlite_db: Path) -> None:
    """Test that tools work correctly with environment variable configuration."""
    # Use the existing temp database to avoid authorizer issues with CREATE statements
    with mock.patch.dict(os.environ, {"SQLITE_DB_PATH": temp_sqlite_db.as_posix()}):
        # Test RunSQLTool with env config
        tool = RunSQLTool()
        
        # Now test the tool with existing data
        result = tool._run(
            test_context,
            query="SELECT * FROM users LIMIT 1"
        )
        rows = result.get_value()
        assert len(rows) == 1
        assert "name" in rows[0]
        assert "age" in rows[0]
