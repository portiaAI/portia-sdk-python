---
sidebar_label: storage
title: portia.storage
---

Storage classes.

## PlanStorage Objects

```python
class PlanStorage(ABC)
```

Base class for storing plans.

#### save\_plan

```python
@abstractmethod
def save_plan(plan: Plan) -> None
```

Save a plan.

#### get\_plan

```python
@abstractmethod
def get_plan(plan_id: UUID) -> Plan
```

Retrieve a plan by its ID.

## WorkflowStorage Objects

```python
class WorkflowStorage(ABC)
```

Base class for storing plans.

#### save\_workflow

```python
@abstractmethod
def save_workflow(workflow: Workflow) -> None
```

Save a workflow.

#### get\_workflow

```python
@abstractmethod
def get_workflow(workflow_id: UUID) -> Workflow
```

Retrieve a workflow by its ID.

## Storage Objects

```python
class Storage(PlanStorage, WorkflowStorage)
```

Combined base class for Plan + Workflow storage.

## InMemoryStorage Objects

```python
class InMemoryStorage(Storage)
```

Simple storage class that keeps plans + workflows in memory.

#### save\_plan

```python
def save_plan(plan: Plan) -> None
```

Add plan to dict.

#### get\_plan

```python
def get_plan(plan_id: UUID) -> Plan
```

Get plan from dict.

#### save\_workflow

```python
def save_workflow(workflow: Workflow) -> None
```

Add workflow to dict.

#### get\_workflow

```python
def get_workflow(workflow_id: UUID) -> Workflow
```

Get workflow from dict.

## DiskFileStorage Objects

```python
class DiskFileStorage(Storage)
```

Disk-based implementation of the Storage interface.

Stores serialized Plan and Workflow objects as JSON files on disk.

#### save\_plan

```python
def save_plan(plan: Plan) -> None
```

Save a Plan object to the storage.

Args:
    plan (Plan): The Plan object to save.

#### get\_plan

```python
def get_plan(plan_id: UUID) -> Plan
```

Retrieve a Plan object by its ID.

Args:
    plan_id (UUID): The ID of the Plan to retrieve.

Returns:
    Plan: The retrieved Plan object.

Raises:
    PlanNotFoundError: If the Plan is not found or validation fails.

#### save\_workflow

```python
def save_workflow(workflow: Workflow) -> None
```

Save a Workflow object to the storage.

Args:
    workflow (Workflow): The Workflow object to save.

#### get\_workflow

```python
def get_workflow(workflow_id: UUID) -> Workflow
```

Retrieve a Workflow object by its ID.

Args:
    workflow_id (UUID): The ID of the Workflow to retrieve.

Returns:
    Workflow: The retrieved Workflow object.

Raises:
    WorkflowNotFoundError: If the Workflow is not found or validation fails.

## PortiaCloudStorage Objects

```python
class PortiaCloudStorage(Storage)
```

Save plans and workflows to portia cloud.

#### save\_plan

```python
def save_plan(plan: Plan) -> None
```

Add plan to cloud.

#### get\_plan

```python
def get_plan(plan_id: UUID) -> Plan
```

Get plan from cloud.

#### save\_workflow

```python
def save_workflow(workflow: Workflow) -> None
```

Add workflow to cloud.

#### get\_workflow

```python
def get_workflow(workflow_id: UUID) -> Workflow
```

Get workflow from cloud.

