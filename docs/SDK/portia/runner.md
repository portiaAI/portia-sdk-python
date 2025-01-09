---
sidebar_label: runner
title: portia.runner
---

Runner classes which actually plan + run queries.

## Runner Objects

```python
class Runner()
```

Create and run plans for queries.

#### run\_query

```python
def run_query(query: str,
              tools: list[Tool] | list[str] | None = None,
              example_workflows: list[Plan] | None = None) -> Workflow
```

Plan and run a query in one go.

#### plan\_query

```python
def plan_query(query: str,
               tools: list[Tool] | list[str] | None = None,
               example_plans: list[Plan] | None = None) -> Plan
```

Plans how to do the query given the set of tools and any examples.

#### run\_plan

```python
def run_plan(plan: Plan) -> Workflow
```

Run a plan returning the completed workflow or clarifications if needed.

#### resume\_workflow

```python
def resume_workflow(workflow: Workflow) -> Workflow
```

Resume a workflow after an interruption.

