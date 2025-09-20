# Migration Guide

This document outlines breaking changes and migration steps for the Portia SDK.

## Version 1.0.0 - Major API Surface Rename

**Release Date:** TBD

### Breaking Changes

#### API Surface Rename (V2 → Main API)

The V2 classes (`PlanV2`, `StepV2`, `PlanBuilderV2`) have been promoted to the main API and are now the default classes without the `V2` suffix. The original V1 classes have been deprecated and removed from the public API.

**Class Renames:**
- `PlanV2` → `Plan`
- `StepV2` → `Step`
- `PlanBuilderV2` → `PlanBuilder`

**Import Changes:**
- `from portia import PlanV2` → `from portia import Plan`
- `from portia import StepV2` → `from portia import Step`
- `from portia import PlanBuilderV2` → `from portia import PlanBuilder`
- `from portia.builder.plan_v2 import PlanV2` → `from portia.plan import Plan`
- `from portia.builder.step_v2 import StepV2` → `from portia.builder.step import Step`
- `from portia.builder.plan_builder_v2 import PlanBuilderV2` → `from portia.builder.plan_builder import PlanBuilder`

**File Structure Changes:**
- `portia/builder/plan_v2.py` → `portia/plan.py`
- `portia/builder/step_v2.py` → `portia/builder/step.py`
- `portia/builder/plan_builder_v2.py` → `portia/builder/plan_builder.py`

### Migration Steps

#### 1. Update Imports

**Before (v0.x):**
```python
from portia import PlanV2, StepV2, PlanBuilderV2
from portia.builder.plan_v2 import PlanV2
from portia.builder.step_v2 import StepV2
from portia.builder.plan_builder_v2 import PlanBuilderV2
```

**After (v1.0.0):**
```python
from portia import Plan, Step, PlanBuilder
from portia.plan import Plan
from portia.builder.step import Step
from portia.builder.plan_builder import PlanBuilder
```

#### 2. Update Class References

Replace all instances of the V2 class names in your code:

**Before (v0.x):**
```python
# Creating a plan builder
builder = PlanBuilderV2()

# Type hints
def process_plan(plan: PlanV2) -> None:
    pass

def create_step() -> StepV2:
    pass

# Variable names (optional but recommended)
my_plan_v2 = builder.build()
```

**After (v1.0.0):**
```python
# Creating a plan builder
builder = PlanBuilder()

# Type hints
def process_plan(plan: Plan) -> None:
    pass

def create_step() -> Step:
    pass

# Variable names (optional but recommended)
my_plan = builder.build()
```

#### 3. Update Documentation and Comments

Replace references to V2 classes in your documentation, comments, and docstrings:

**Before (v0.x):**
```python
# Build a PlanV2 using PlanBuilderV2
# Each StepV2 represents an action...
```

**After (v1.0.0):**
```python
# Build a Plan using PlanBuilder
# Each Step represents an action...
```

### Deprecated and Removed Features

#### Removed from Public API

The following V1 classes are no longer exported from the public API:

- `Plan` (V1) - Use `Plan` (formerly `PlanV2`) instead
- `Step` (V1) - Use `Step` (formerly `StepV2`) instead
- `PlanBuilder` (V1) - Use `PlanBuilder` (formerly `PlanBuilderV2`) instead

These classes may still exist internally for backward compatibility but should not be used in new code.

#### Removed Exports

The following exports have been removed from `portia.__all__`:

- `PlanV2` (renamed to `Plan`)
- `StepV2` (renamed to `Step`)
- `PlanBuilderV2` (renamed to `PlanBuilder`)

### Why This Change?

This change simplifies the API by:

1. **Removing version suffixes**: Users no longer need to remember to use "V2" classes
2. **Cleaner imports**: More intuitive import statements without version numbers
3. **Unified API**: Single set of classes for plan building and execution
4. **Future-proofing**: Eliminates the need for further V3, V4, etc. versioning

### Automatic Migration Tools

Consider using the following find-and-replace patterns to help with migration:

```bash
# Update imports
find . -name "*.py" -exec sed -i 's/from portia import PlanV2/from portia import Plan/g' {} \;
find . -name "*.py" -exec sed -i 's/from portia import StepV2/from portia import Step/g' {} \;
find . -name "*.py" -exec sed -i 's/from portia import PlanBuilderV2/from portia import PlanBuilder/g' {} \;

# Update specific imports
find . -name "*.py" -exec sed -i 's/from portia.builder.plan_v2 import PlanV2/from portia.plan import Plan/g' {} \;
find . -name "*.py" -exec sed -i 's/from portia.builder.step_v2 import StepV2/from portia.builder.step import Step/g' {} \;
find . -name "*.py" -exec sed -i 's/from portia.builder.plan_builder_v2 import PlanBuilderV2/from portia.builder.plan_builder import PlanBuilder/g' {} \;

# Update class names (be careful with these, review changes)
find . -name "*.py" -exec sed -i 's/PlanV2/Plan/g' {} \;
find . -name "*.py" -exec sed -i 's/StepV2/Step/g' {} \;
find . -name "*.py" -exec sed -i 's/PlanBuilderV2/PlanBuilder/g' {} \;
```

**Note:** Always review the changes made by these commands, as they may affect comments or string literals that should not be changed.

### Need Help?

If you encounter issues during migration:

1. Check the [documentation](https://docs.portialabs.ai) for updated examples
2. Review the [example files](./example_builder.py) in this repository
3. Open an issue on the [GitHub repository](https://github.com/portiaAI/portia-sdk-python)

### Timeline

- **v0.8.x**: Last version with V2 classes, deprecation warnings added
- **v1.0.0**: V2 classes renamed to main API, V1 classes removed from public API
- **Future versions**: V1 classes may be completely removed from codebase