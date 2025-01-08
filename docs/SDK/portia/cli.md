---
sidebar_label: cli
title: portia.cli
---

CLI Implementation.

Usage:

portia-cli run &quot;add 4 + 8&quot; - run a query
portia-cli plan &quot;add 4 + 8&quot; - plan a query

#### cli

```python
@click.group()
def cli() -> None
```

Portia CLI.

#### run

```python
@click.command()
@click.argument("query")
def run(query: str) -> None
```

Run a query.

#### plan

```python
@click.command()
@click.argument("query")
def plan(query: str) -> None
```

Plan a query.

