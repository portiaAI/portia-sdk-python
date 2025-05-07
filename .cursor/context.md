# Portia SDK Python

## Project Overview
This is an open-source framework for creating reliable and production-ready multi-agent systems.

## Core Architecture
Portia implements a three-agent architecture to ensure robust and reliable execution:

1. **Planning Agent**: Creates a comprehensive plan for how to achieve a given task, breaking it down into executable steps.
2. **Execution Agent**: Executes individual steps of the plan, focusing on the implementation details and concrete actions required.
3. **Introspection Agent**: Operates between execution steps to check which step is needed next.

## Developing

You can run linting in the codebase by running the following commands, but only do this if asked:
* poetry run pyright
* poetry run ruff check --fix

If this doesn't work, you may need to install poetry with pip install poetry