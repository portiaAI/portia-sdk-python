"""Entry point for running portia modules as scripts.

This allows running portia modules with `python -m portia.<module_name>`.
Currently supports:
- migrate_legacy_plans: Migration script for legacy plans
"""

import sys
from pathlib import Path


def main() -> None:
    """Main entry point for module execution."""
    if len(sys.argv) < 2:
        print("Usage: python -m portia <subcommand>")
        print("Available subcommands:")
        print("  migrate_legacy_plans - Migrate legacy plans to JSON archive format")
        sys.exit(1)

    subcommand = sys.argv[1]

    if subcommand == "migrate_legacy_plans":
        from portia.migrate_legacy_plans import main as migrate_main
        # The migration script expects to be called directly, so we adjust sys.argv
        # to simulate direct execution
        original_argv = sys.argv[:]
        try:
            sys.argv = [f"python -m portia.migrate_legacy_plans"] + sys.argv[2:]
            result = migrate_main()
            sys.exit(result)
        finally:
            sys.argv = original_argv
    else:
        print(f"Unknown subcommand: {subcommand}")
        print("Available subcommands:")
        print("  migrate_legacy_plans - Migrate legacy plans to JSON archive format")
        sys.exit(1)


if __name__ == "__main__":
    main()