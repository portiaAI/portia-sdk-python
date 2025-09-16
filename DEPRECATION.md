# Plan V1 Deprecation and Feature Flags

This document describes the deprecation of Plan V1 classes and the feature flags introduced to support the transition to Plan V2.

## Deprecated Classes

The following classes are deprecated as of version 0.8.0:

### `PlanBuilder` (deprecated)
- **Replacement**: `PlanBuilderV2`
- **Reason**: The new builder provides better type safety, improved API design, and enhanced functionality
- **Warnings**: Emits deprecation warnings at both import-time and initialization-time

### `Plan` (deprecated)
- **Replacement**: `PlanV2`
- **Reason**: The new plan structure provides better validation, improved serialization, and enhanced features
- **Warnings**: Emits deprecation warnings at both import-time and initialization-time

## Feature Flags

### `PLAN_V2_DEFAULT`

This environment variable controls the default behavior for Plan V2 adoption.

**Environment Variable**: `PLAN_V2_DEFAULT`
**Config Flag**: `plan_v2_default`
**Default Value**: `false`

#### Usage

```bash
# Enable Plan V2 as default
export PLAN_V2_DEFAULT=true

# Disable Plan V2 as default (explicit)
export PLAN_V2_DEFAULT=false
```

#### Configuration

The feature flag is automatically loaded into the Portia configuration:

```python
from portia import Config
from portia.config import FEATURE_FLAG_PLAN_V2_DEFAULT

config = Config.from_default()
is_plan_v2_default = config.feature_flags[FEATURE_FLAG_PLAN_V2_DEFAULT]
```

#### Values

The flag accepts the following values (case-insensitive):
- `"true"`, `"TRUE"`, `"True"` → `True`
- `"false"`, `"FALSE"`, `"False"` → `False`
- Any other value → `False` (default)

## Migration Guide

### For Library Users

1. **Update Imports**: Replace imports of `Plan` and `PlanBuilder` with `PlanV2` and `PlanBuilderV2`
   ```python
   # Before
   from portia import Plan, PlanBuilder

   # After
   from portia import PlanV2, PlanBuilderV2
   ```

2. **Update Code**: Replace usage of deprecated classes
   ```python
   # Before
   builder = PlanBuilder("My query")
   builder.step("Do something", output="$result")
   plan = builder.build()

   # After
   builder = PlanBuilderV2("My query")
   builder.llm_step("Do something", output_name="result")
   plan = builder.build()
   ```

3. **Set Feature Flag** (optional): Enable Plan V2 as default behavior
   ```bash
   export PLAN_V2_DEFAULT=true
   ```

### For Library Maintainers

The deprecation system provides both import-time and initialization-time warnings:

- **Import-time warnings**: Emitted when deprecated classes are imported from the main module
- **Initialization-time warnings**: Emitted when deprecated class instances are created
- **Stacklevel handling**: Warnings point to user code, not internal deprecation infrastructure

## Testing

The deprecation system includes comprehensive tests covering:

- Feature flag behavior with various environment variable values
- Deprecation warnings for class initialization
- Import-time deprecation warnings
- Backwards compatibility preservation
- Integration with existing configuration system

## Timeline

- **v0.8.0**: Deprecation warnings introduced
- **Future version**: Plan V1 classes will be removed (specific version TBD)

Users are encouraged to migrate to Plan V2 classes as soon as possible to ensure compatibility with future versions.