"""SQL Tools with pluggable adapter and a default SQLite implementation.

This module provides separate tools for different SQL operations against a database
via a pluggable adapter. By default, it ships with a SQLite adapter that can be
configured via environment variables or pure JSON config passed at call time.

Available tools:
- RunSQLTool: Execute read-only SQL SELECT queries
- ListTablesTool: List all available tables in the database
- GetTableSchemasTool: Get detailed schema information for specified tables
- CheckSQLTool: Validate SQL queries without executing them

Legacy SQLTool is also available for backward compatibility but is deprecated.

Security note: Only read-only operations are allowed. SQLite authorizer is used to enforce
read-only access by denying all write operations (INSERT, UPDATE, DELETE, CREATE, etc.).
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from portia.errors import ToolHardError, ToolSoftError
from portia.tool import Tool, ToolRunContext


def _sqlite_authorizer(
    action: int,
    arg1: str | None,
    _arg2: str | None,
    _db_name: str | None,
    _trigger_or_view: str | None,
) -> int:
    """SQLite authorizer function that only allows SELECT operations.

    Args:
        action: The SQLite action code (e.g., sqlite3.SQLITE_READ, sqlite3.SQLITE_SELECT)
        arg1: First argument depending on the action (e.g., table name, column name)
        arg2: Second argument depending on the action (e.g., column name)
        db_name: Database name
        trigger_or_view: Name of trigger or view, if applicable

    Returns:
        sqlite3.SQLITE_OK for allowed operations, sqlite3.SQLITE_DENY for forbidden ones

    """
    # Allow read operations
    if action in (
        sqlite3.SQLITE_READ,  # Reading data from tables
        sqlite3.SQLITE_SELECT,  # SELECT statements
        sqlite3.SQLITE_FUNCTION,  # Using functions in queries
    ):
        return sqlite3.SQLITE_OK
    # Allow only specific safe, read-only PRAGMA statements
    if action == sqlite3.SQLITE_PRAGMA:
        pragma_name = arg1.lower() if arg1 else ""
        # Only allow read-only PRAGMA operations
        safe_pragmas = {
            "table_info",  # Get table schema information
            "table_list",  # List tables
            "database_list",  # List databases
            "foreign_key_list",  # List foreign keys
            "index_list",  # List indexes
            "index_info",  # Get index information
            "schema_version",  # Get schema version
            "user_version",  # Get user version (read-only)
            "compile_options",  # Get compile options
        }
        if pragma_name in safe_pragmas:
            return sqlite3.SQLITE_OK
        # Deny potentially dangerous PRAGMA operations
        return sqlite3.SQLITE_DENY
    # Deny all other operations (INSERT, UPDATE, DELETE, CREATE, etc.)
    return sqlite3.SQLITE_DENY


class SQLAdapter:
    """Abstract adapter interface for SQL databases (read-only)."""

    def run_sql(self, query: str) -> list[dict[str, Any]]:
        """Execute a read-only query and return rows as list of dicts."""
        raise NotImplementedError

    def list_tables(self) -> list[str]:
        """List available table names."""
        raise NotImplementedError

    def get_table_schemas(self, tables: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Return column schemas for the given tables."""
        raise NotImplementedError

    def check_sql(self, query: str) -> dict[str, Any]:
        """Check if a query would run successfully (read-only)."""
        raise NotImplementedError


@dataclass
class SQLiteConfig:
    """Configuration for SQLite connections.

    Attributes:
        db_path: Path to the SQLite database file, or ":memory:" for in-memory.

    """

    db_path: str


class SQLiteAdapter(SQLAdapter):
    """SQLite adapter using read-only URI mode where possible."""

    def __init__(self, config: SQLiteConfig) -> None:
        """Initialize the adapter with the given configuration."""
        self.config = config

    def _connect(self) -> sqlite3.Connection:
        """Open a connection to the database with read-only authorizer."""
        if self.config.db_path == ":memory:":
            conn = sqlite3.connect(self.config.db_path)
        else:
            uri = f"file:{self.config.db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)

        # Set the authorizer to enforce read-only operations
        conn.set_authorizer(_sqlite_authorizer)
        conn.row_factory = sqlite3.Row
        return conn

    def run_sql(self, query: str) -> list[dict[str, Any]]:
        """Execute the read-only SQL query and return rows as dicts."""
        try:
            with self._connect() as conn:
                cur = conn.execute(query)
                rows = cur.fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as e:
            raise ToolHardError(f"SQLite error: {e}") from e

    def list_tables(self) -> list[str]:
        """Return a list of user tables in the database."""
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                    "ORDER BY name"
                )
                return [r[0] for r in cur.fetchall()]
        except sqlite3.Error as e:
            raise ToolHardError(f"SQLite error: {e}") from e

    def get_table_schemas(self, tables: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Return PRAGMA table_info for each table in `tables`."""
        out: dict[str, list[dict[str, Any]]] = {}
        try:
            with self._connect() as conn:
                for t in tables:
                    cur = conn.execute(f"PRAGMA table_info({t})")
                    cols = cur.fetchall()
                    out[t] = [
                        {
                            "cid": c[0],
                            "name": c[1],
                            "type": c[2],
                            "notnull": c[3],
                            "dflt_value": c[4],
                            "pk": c[5],
                        }
                        for c in cols
                    ]
        except sqlite3.Error as e:
            raise ToolHardError(f"SQLite error: {e}") from e
        else:
            return out

    def check_sql(self, query: str) -> dict[str, Any]:
        """Check the query by executing an EXPLAIN; return ok True/False with error."""
        try:
            with self._connect() as conn:
                conn.execute(f"EXPLAIN {query}")
        except sqlite3.Error as e:
            return {"ok": False, "error": str(e)}
        else:
            return {"ok": True}


class BaseSQLToolArgs(BaseModel):
    """Base arguments for SQL tools.

    Either provide config via:
      - environment variables, or
      - the optional `config_json` string with adapter-specific config (pure JSON)
    """

    config_json: str | None = Field(
        default=None,
        description=(
            'Adapter configuration as pure JSON string (e.g., {"db_path": "/tmp/db.sqlite"})'
        ),
    )


class BaseSQLTool(Tool[Any]):
    """Base SQL tool with shared adapter functionality.

    Use SQLiteAdapter by default. Configure via env or config_json:
      - SQLITE_DB_PATH: path to sqlite database (e.g., /tmp/db.sqlite).
        If not set, defaults to :memory:
    """

    output_schema: tuple[str, str] = ("json", "JSON result for the SQL operation")

    # Use a private attribute to avoid Pydantic BaseModel field restrictions
    _adapter: SQLAdapter | None = PrivateAttr(default=None)

    def __init__(self, adapter: SQLAdapter | None = None, **kwargs: Any) -> None:
        """Initialize the tool with an optional adapter (defaults to SQLite)."""
        # Let subclasses handle their own initialization with their class attributes
        super().__init__(**kwargs)
        self._adapter = adapter

    def _adapter_from_env(self) -> SQLAdapter:
        """Create an adapter from environment variables (SQLite only for now)."""
        db_path = os.getenv("SQLITE_DB_PATH", ":memory:")
        return SQLiteAdapter(SQLiteConfig(db_path=db_path))

    @staticmethod
    def _adapter_from_config_json(config_json: str | None) -> SQLAdapter | None:
        """Create an adapter from a JSON config string, if provided."""
        if not config_json:
            return None
        try:
            cfg = json.loads(config_json)
        except json.JSONDecodeError as e:
            raise ToolSoftError(f"Invalid config_json: {e}") from e
        db_path = cfg.get("db_path")
        if not isinstance(db_path, str) or not db_path:
            raise ToolSoftError("config_json must include a non-empty 'db_path' for SQLite")
        return SQLiteAdapter(SQLiteConfig(db_path=db_path))

    def _get_adapter(self, config_json: str | None = None) -> SQLAdapter:
        """Get the appropriate adapter based on config."""
        return (
            self._adapter_from_config_json(config_json) or self._adapter or self._adapter_from_env()
        )


class RunSQLArgs(BaseSQLToolArgs):
    """Arguments for running SQL queries."""

    query: str = Field(..., description="SQL query to execute (SELECT only)")


class RunSQLTool(BaseSQLTool):
    """Execute read-only SQL SELECT queries against a database."""

    id: str = "run_sql"
    name: str = "Run SQL Query"
    description: str = (
        "Execute read-only SQL SELECT queries against a database. Only SELECT queries are "
        "allowed. Configuration: Set SQLITE_DB_PATH env var or provide config_json parameter. "
        'config_json format: {"db_path": "/path/to/database.db"} where db_path is the SQLite '
        'database file path. Use ":memory:" for in-memory database. Default: :memory: if no '
        "config provided."
    )
    args_schema: type[BaseModel] = RunSQLArgs

    def run(self, _: ToolRunContext, **kwargs: Any) -> list[dict[str, Any]]:
        """Execute the SQL query and return results."""
        args = RunSQLArgs.model_validate(kwargs)
        adapter = self._get_adapter(args.config_json)
        return adapter.run_sql(args.query)


class ListTablesArgs(BaseSQLToolArgs):
    """Arguments for listing database tables."""

    # Only needs base config args


class ListTablesTool(BaseSQLTool):
    """List all available tables in the database."""

    id: str = "list_tables"
    name: str = "List Database Tables"
    description: str = (
        "List all available tables in the database. Configuration: Set SQLITE_DB_PATH env var "
        'or provide config_json parameter. config_json format: {"db_path": "/path/to/database.db"} '
        'where db_path is the SQLite database file path. Use ":memory:" for in-memory database. '
        "Default: :memory: if no config provided."
    )
    args_schema: type[BaseModel] = ListTablesArgs

    def run(self, _: ToolRunContext, **kwargs: Any) -> list[str]:
        """List all tables in the database."""
        args = ListTablesArgs.model_validate(kwargs)
        adapter = self._get_adapter(args.config_json)
        return adapter.list_tables()


class GetTableSchemasArgs(BaseSQLToolArgs):
    """Arguments for getting table schemas."""

    tables: list[str] = Field(..., description="List of table names to get schemas for")


class GetTableSchemasTool(BaseSQLTool):
    """Get detailed schema information for specified tables."""

    id: str = "get_table_schemas"
    name: str = "Get Table Schemas"
    description: str = (
        "Get detailed schema information (columns, types, etc.) for specified tables. "
        "Returns column details including name, type, nullability, and primary key info. "
        "Configuration: Set SQLITE_DB_PATH env var or provide config_json parameter. "
        'config_json format: {"db_path": "/path/to/database.db"} where db_path is the '
        'SQLite database file path. Use ":memory:" for in-memory database. '
        "Default: :memory: if no config provided."
    )

    args_schema: type[BaseModel] = GetTableSchemasArgs

    def run(self, _: ToolRunContext, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        """Get schema information for the specified tables."""
        args = GetTableSchemasArgs.model_validate(kwargs)
        adapter = self._get_adapter(args.config_json)
        return adapter.get_table_schemas(args.tables)


class CheckSQLArgs(BaseSQLToolArgs):
    """Arguments for checking SQL query validity."""

    query: str = Field(..., description="SQL query to validate (SELECT only)")


class CheckSQLTool(BaseSQLTool):
    """Check if a SQL query is valid without executing it."""

    id: str = "check_sql"
    name: str = "Check SQL Query"
    description: str = (
        "Check if a SQL query is valid without executing it. Uses EXPLAIN to validate "
        "syntax and table/column references. Only SELECT queries are allowed. Returns "
        "{ok: true} for valid queries or {ok: false, error: 'message'} for invalid ones. "
        "Configuration: Set SQLITE_DB_PATH env var or provide config_json parameter. "
        'config_json format: {"db_path": "/path/to/database.db"} where db_path is the '
        'SQLite database file path. Use ":memory:" for in-memory database. Default: '
        ":memory: if no config provided."
    )
    args_schema: type[BaseModel] = CheckSQLArgs

    def run(self, _: ToolRunContext, **kwargs: Any) -> dict[str, Any]:
        """Check the validity of the SQL query."""
        args = CheckSQLArgs.model_validate(kwargs)
        adapter = self._get_adapter(args.config_json)
        return adapter.check_sql(args.query)
