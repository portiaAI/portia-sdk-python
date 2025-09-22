#!/usr/bin/env python3
"""Migration script for legacy plans.

This script helps users migrate their legacy plans from old storage formats to archival JSON files.
It enumerates all legacy plans found in storage and exports them for manual migration.

Usage:
    python -m portia.migrate_legacy_plans [--storage-dir <dir>] [--output-dir <dir>] [--dry-run]

Options:
    --storage-dir <dir>    Directory containing legacy plans (default: .portia)
    --output-dir <dir>     Directory to export JSON files (default: ./legacy_plans_export)
    --dry-run             Show what would be migrated without actually exporting
    --help                Show this help message

The script will:
1. Scan the storage directory for legacy plan files
2. Parse and validate each legacy plan
3. Export valid plans to JSON files with metadata
4. Generate a migration report with statistics and any errors encountered
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from portia.plan import Plan
from portia.prefixed_uuid import PLAN_UUID_PREFIX


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_legacy_plan_files(storage_dir: Path) -> list[Path]:
    """Find all legacy plan files in the storage directory.

    Args:
        storage_dir: Path to the storage directory to scan

    Returns:
        List of Path objects pointing to legacy plan files
    """
    if not storage_dir.exists():
        logging.warning(f"Storage directory {storage_dir} does not exist")
        return []

    # Look for files that start with the plan UUID prefix
    plan_files = []
    for file_path in storage_dir.iterdir():
        if file_path.is_file() and file_path.name.startswith(PLAN_UUID_PREFIX) and file_path.name.endswith('.json'):
            plan_files.append(file_path)

    logging.info(f"Found {len(plan_files)} potential legacy plan files")
    return plan_files


def load_legacy_plan(file_path: Path) -> tuple[Plan | None, str | None]:
    """Attempt to load a legacy plan from a file.

    Args:
        file_path: Path to the legacy plan file

    Returns:
        Tuple of (Plan object or None, error message or None)
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            plan_data = json.load(f)

        # Try to parse as a Plan object
        plan = Plan.model_validate(plan_data)
        return plan, None
    except Exception as e:
        return None, str(e)


def export_plan_to_json(plan: Plan, output_dir: Path, file_name: str) -> Path:
    """Export a plan to a JSON file with metadata.

    Args:
        plan: The Plan object to export
        output_dir: Directory to save the exported file
        file_name: Name of the output file

    Returns:
        Path to the exported file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create export data with metadata
    export_data = {
        "migration_metadata": {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "migrated_from": "legacy_storage",
            "plan_id": str(plan.id),
            "original_query": plan.plan_context.query,
        },
        "plan_data": plan.model_dump(mode="json"),
    }

    output_file = output_dir / file_name
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    return output_file


def generate_migration_report(
    total_files: int,
    successful_exports: int,
    failed_files: list[tuple[Path, str]],
    output_dir: Path,
) -> Path:
    """Generate a migration report with statistics and errors.

    Args:
        total_files: Total number of files processed
        successful_exports: Number of successful exports
        failed_files: List of (file_path, error_message) tuples for failed files
        output_dir: Directory to save the report

    Returns:
        Path to the generated report file
    """
    report_data = {
        "migration_summary": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_files_found": total_files,
            "successful_exports": successful_exports,
            "failed_files": len(failed_files),
            "success_rate": f"{(successful_exports / total_files * 100):.1f}%" if total_files > 0 else "0.0%",
        },
        "failed_files": [
            {
                "file_path": str(file_path),
                "error_message": error_msg,
            }
            for file_path, error_msg in failed_files
        ],
    }

    report_file = output_dir / "migration_report.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    return report_file


def main() -> int:
    """Main migration script entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate legacy Portia plans to JSON archive format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path(".portia"),
        help="Directory containing legacy plans (default: .portia)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./legacy_plans_export"),
        help="Directory to export JSON files (default: ./legacy_plans_export)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually exporting",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logging.info("Starting legacy plan migration")
    logging.info(f"Storage directory: {args.storage_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Dry run mode: {args.dry_run}")

    # Find all legacy plan files
    plan_files = find_legacy_plan_files(args.storage_dir)

    if not plan_files:
        logging.info("No legacy plan files found")
        return 0

    successful_exports = 0
    failed_files: list[tuple[Path, str]] = []

    # Process each file
    for file_path in plan_files:
        logging.debug(f"Processing file: {file_path}")

        # Attempt to load the plan
        plan, error = load_legacy_plan(file_path)

        if plan is None:
            logging.warning(f"Failed to load plan from {file_path}: {error}")
            failed_files.append((file_path, error or "Unknown error"))
            continue

        # Export the plan if not in dry-run mode
        if not args.dry_run:
            try:
                export_file = export_plan_to_json(
                    plan,
                    args.output_dir,
                    f"migrated_{file_path.name}",
                )
                logging.info(f"Exported plan {plan.id} to {export_file}")
                successful_exports += 1
            except Exception as e:
                logging.error(f"Failed to export plan {plan.id}: {e}")
                failed_files.append((file_path, f"Export failed: {e}"))
        else:
            logging.info(f"Would export plan {plan.id} from {file_path}")
            successful_exports += 1

    # Generate migration report
    if not args.dry_run:
        report_file = generate_migration_report(
            len(plan_files),
            successful_exports,
            failed_files,
            args.output_dir,
        )
        logging.info(f"Migration report saved to {report_file}")

    # Print summary
    print("\n" + "="*60)
    print("LEGACY PLAN MIGRATION SUMMARY")
    print("="*60)
    print(f"Total files found: {len(plan_files)}")
    print(f"Successfully processed: {successful_exports}")
    print(f"Failed to process: {len(failed_files)}")
    print(f"Success rate: {(successful_exports / len(plan_files) * 100):.1f}%" if plan_files else "0.0%")

    if failed_files:
        print(f"\nFailed files:")
        for file_path, error in failed_files:
            print(f"  {file_path}: {error}")

    if not args.dry_run and successful_exports > 0:
        print(f"\nExported plans are saved in: {args.output_dir}")
        print("You can now recreate these plans using the current Portia API.")

    return 0 if not failed_files else 1


if __name__ == "__main__":
    sys.exit(main())