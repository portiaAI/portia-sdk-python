"""Generic SQL Tool with pluggable adapter and a default SQLite implementation.

This tool allows read-only querying against an SQL database via a pluggable adapter.
By default, it ships with a SQLite adapter that can be configured via environment
variables or pure JSON config passed at call time.

Exposed actions:
- run_sql(query: str)
- list_tables()
- get_table_schemas(tables: list[str])
- check_sql(query: str)

Security note: Only read-only operations are allowed. For safety, queries must start with
SELECT (ignoring leading whitespace and comments). Other statements are rejected with a
ToolSoftError.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from portia.errors import ToolHardError, ToolSoftError
from portia.tool import Tool, ToolRunContext

READ_ONLY_SELECT_RE = re.compile(
    r"^\s*(?:/\*.*?\*/\s*)*(?:--.*?$\s*)*select\b",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)


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
        """Open a connection to the database in read-only mode when possible."""
        if self.config.db_path == ":memory:":
            conn = sqlite3.connect(self.config.db_path)
        else:
            uri = f"file:{self.config.db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_read_only(self, query: str) -> None:
        """Ensure the query is a SELECT; otherwise raise a soft error."""
        if not READ_ONLY_SELECT_RE.search(query):
            raise ToolSoftError("Only read-only SELECT queries are allowed")

    def run_sql(self, query: str) -> list[dict[str, Any]]:
        """Execute the read-only SQL query and return rows as dicts."""
        self._ensure_read_only(query)
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
            self._ensure_read_only(query)
            with self._connect() as conn:
                conn.execute(f"EXPLAIN {query}")
        except ToolSoftError:
            raise
        except sqlite3.Error as e:
            return {"ok": False, "error": str(e)}
        else:
            return {"ok": True}


class SQLToolArgs(BaseModel):
    """Arguments for SQLTool actions.

    Either provide config via:
      - environment variables, or
      - the optional `config_json` string with adapter-specific config (pure JSON)
    """

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


class SQLTool(Tool[Any]):
    """Generic SQL tool with pluggable adapter.

    Use SQLiteAdapter by default. Configure via env or config_json:
      - SQLITE_DB_PATH: path to sqlite database (e.g., /tmp/db.sqlite).
        If not set, defaults to :memory:
    """

    id: str = "sql_tool"
    name: str = "SQL Tool"
    description: str = (
        "Run read-only SQL operations through a pluggable adapter. Supported actions: "
        "run_sql, list_tables, get_table_schemas, check_sql. Only SELECT queries are allowed for "
        "run_sql/check_sql. Default adapter is SQLite. Configure via env (SQLITE_DB_PATH) or "
        "pass config_json with adapter parameters."
    )
    args_schema = SQLToolArgs
    output_schema: tuple[str, str] = ("json", "JSON result for the selected action")

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

    def run(self, _: ToolRunContext, **kwargs: Any) -> Any:  
        """Dispatch to the configured adapter based on the requested action."""
        args = SQLToolArgs.model_validate(kwargs)
        adapter = self._adapter_from_config_json(args.config_json) or self._adapter

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
