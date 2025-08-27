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
from typing import Any, Type

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from portia.errors import ToolHardError, ToolSoftError
from portia.tool import Tool, ToolRunContext

def _sqlite_authorizer(action, arg1, arg2, db_name, trigger_or_view):
    """SQLite authorizer function that only allows SELECT operations.
    
    Args:
        action: The SQLite action code (e.g., sqlite3.SQLITE_READ, sqlite3.SQLITE_SELECT)
        arg1, arg2: Additional arguments depending on the action
        db_name: Database name
        trigger_or_view: Name of trigger or view, if applicable
        
    Returns:
        sqlite3.SQLITE_OK for allowed operations, sqlite3.SQLITE_DENY for forbidden ones
    """
    # Allow read operations
    if action in (
        sqlite3.SQLITE_READ,      # Reading data from tables
        sqlite3.SQLITE_SELECT,    # SELECT statements
        sqlite3.SQLITE_FUNCTION,  # Using functions in queries
        sqlite3.SQLITE_PRAGMA,    # PRAGMA statements (for table_info, etc.)
    ):
        return sqlite3.SQLITE_OK
    
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

    def get_table_schemas(
        self, tables: list[str]
    ) -> dict[str, list[dict[str, Any]]]:  
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
    _adapter: SQLAdapter = PrivateAttr()

    def __init__(self, adapter: SQLAdapter | None = None) -> None:
        """Initialize the tool with an optional adapter (defaults to SQLite)."""
        super().__init__()
        self._adapter = adapter or self._adapter_from_env()

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
        return self._adapter_from_config_json(config_json) or self._adapter


class RunSQLArgs(BaseSQLToolArgs):
    """Arguments for running SQL queries."""
    query: str = Field(..., description="SQL query to execute (SELECT only)")


class RunSQLTool(BaseSQLTool):
    """Execute read-only SQL SELECT queries against a database."""

    id: str = "run_sql"
    name: str = "Run SQL Query"
    description: str = (
        "Execute read-only SQL SELECT queries against a database. "
        "Only SELECT queries are allowed. Default adapter is SQLite. "
        "Configure via env (SQLITE_DB_PATH) or pass config_json with adapter parameters."
    )
    args_schema: Type[RunSQLArgs] = RunSQLArgs

    def run(self, _: ToolRunContext, **kwargs: Any) -> Any:
        """Execute the SQL query and return results."""
        args = RunSQLArgs.model_validate(kwargs)
        adapter = self._get_adapter(args.config_json)
        return adapter.run_sql(args.query)


class ListTablesArgs(BaseSQLToolArgs):
    """Arguments for listing database tables."""
    pass  # Only needs base config args


class ListTablesTool(BaseSQLTool):
    """List all available tables in the database."""

    id: str = "list_tables"
    name: str = "List Database Tables"
    description: str = (
        "List all available tables in the database. "
        "Default adapter is SQLite. Configure via env (SQLITE_DB_PATH) or "
        "pass config_json with adapter parameters."
    )
    args_schema: Type[ListTablesArgs] = ListTablesArgs

    def run(self, _: ToolRunContext, **kwargs: Any) -> Any:
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
        "Default adapter is SQLite. Configure via env (SQLITE_DB_PATH) or "
        "pass config_json with adapter parameters."
    )
    args_schema: Type[GetTableSchemasArgs] = GetTableSchemasArgs

    def run(self, _: ToolRunContext, **kwargs: Any) -> Any:
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
        "Check if a SQL query is valid without executing it. Uses EXPLAIN to validate. "
        "Only SELECT queries are allowed. Default adapter is SQLite. "
        "Configure via env (SQLITE_DB_PATH) or pass config_json with adapter parameters."
    )
    args_schema: Type[CheckSQLArgs] = CheckSQLArgs

    def run(self, _: ToolRunContext, **kwargs: Any) -> Any:
        """Check the validity of the SQL query."""
        args = CheckSQLArgs.model_validate(kwargs)
        adapter = self._get_adapter(args.config_json)
        return adapter.check_sql(args.query)


# Legacy class for backward compatibility (deprecated)
class SQLToolArgs(BaseModel):
    """Deprecated: Arguments for the legacy SQLTool. Use specific tools instead."""

    action: str = Field(
        ...,
        description=("Action to perform: run_sql | list_tables | get_table_schemas | check_sql"),
    )
    query: str | None = Field(default=None, description="SQL query for run_sql/check_sql")
    tables: list[str] | None = Field(default=None, description="Tables for get_table_schemas")
    config_json: str | None = Field(
        default=None,
        description=(
            'Adapter configuration as pure JSON string (e.g., {"db_path": "/tmp/db.sqlite"})'
        ),
    )

    @model_validator(mode="after")
    def _validate_fields(self) -> SQLToolArgs:
        match self.action:
            case "run_sql" | "check_sql":
                if not self.query:
                    raise ValueError("'query' is required for this action")
            case "get_table_schemas":
                if not self.tables:
                    raise ValueError("'tables' is required for this action")
            case "list_tables":
                pass
            case _:
                raise ValueError("Unsupported action")
        return self


class SQLTool(BaseSQLTool):
    """Deprecated: Generic SQL tool with actions. Use specific tools instead.

    This class is kept for backward compatibility. New code should use:
    - RunSQLTool for executing queries
    - ListTablesTool for listing tables  
    - GetTableSchemasTool for getting schemas
    - CheckSQLTool for validating queries
    """

    id: str = "sql_tool"
    name: str = "SQL Tool (Deprecated)"
    description: str = (
        "DEPRECATED: Use specific tools instead (run_sql, list_tables, get_table_schemas, check_sql). "
        "Run read-only SQL operations through a pluggable adapter. Only SELECT queries are allowed. "
        "Default adapter is SQLite. Configure via env (SQLITE_DB_PATH) or pass config_json with adapter parameters."
    )
    args_schema: Type[SQLToolArgs] = SQLToolArgs

    def run(self, _: ToolRunContext, **kwargs: Any) -> Any:  
        """Dispatch to the configured adapter based on the requested action."""
        args = SQLToolArgs.model_validate(kwargs)
        adapter = self._get_adapter(args.config_json)

        match args.action:
            case "run_sql":
                return adapter.run_sql(args.query or "")
            case "list_tables":
                return adapter.list_tables()
            case "get_table_schemas":
                return adapter.get_table_schemas(args.tables or [])
            case "check_sql":
                return adapter.check_sql(args.query or "")
            case _:  
                raise ToolSoftError("Unsupported action")
