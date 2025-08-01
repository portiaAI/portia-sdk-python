# Portia SDK Python

## Project Overview
This is an open-source framework for creating reliable and production-ready multi-agent systems.

## Core Architecture
Portia implements a three-agent architecture to ensure robust and reliable execution:

1. **Planning Agent**: Creates a comprehensive plan for how to achieve a given task, breaking it down into executable steps.
2. **Execution Agent**: Executes individual steps of the plan, focusing on the implementation details and concrete actions required.
3. **Introspection Agent**: Operates between execution steps to check which step is needed next.

The Portia interface has both synchronous and asynchronous methods, typically denoted with an `a` prefix, eg portia.plan() -> portia.aplan(). These should be kept in line with each other, so if you make changes to one, you should make similar changes to the other. 

## Developing

You can run linting in the codebase by running the following commands, but only do this if asked:
* uv run pyright
* uv run ruff check --fix
* uv run pytest {command} -n auto
All python commands should be run with `uv run` and python should never be called directly for anything. 

If this doesn't work, you may need to install uv with pip install uv, making sure to run `uv sync --all-extras --all-groups` after to ensure all dependencies are installed.