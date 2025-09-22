"""Tests for legacy plan migration and deprecation handling."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest

from portia.config import Config, StorageClass
from portia.errors import LegacyPlanDeprecationError, PlanNotFoundError
from portia.migrate_legacy_plans import (
    export_plan_to_json,
    find_legacy_plan_files,
    generate_migration_report,
    load_legacy_plan,
    main as migrate_main,
)
from portia.plan import Plan, PlanContext, PlanUUID, Step
from portia.portia import Portia
from portia.storage import DiskFileStorage


class TestLegacyPlanDeprecationError:
    """Test the LegacyPlanDeprecationError class."""

    def test_error_message_contains_migration_instructions(self) -> None:
        """Test that the error message includes migration instructions."""
        plan_id = PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678")
        error = LegacyPlanDeprecationError(plan_id)

        error_message = str(error)
        assert str(plan_id) in error_message
        assert "python -m portia.migrate_legacy_plans" in error_message
        assert "deprecated" in error_message.lower()
        assert "migration" in error_message.lower()


class TestPortiaLegacyPlanDetection:
    """Test legacy plan detection in Portia."""

    def test_is_legacy_plan_with_disk_storage_and_existing_file(self) -> None:
        """Test _is_legacy_plan returns True when a legacy plan file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            plan_id = PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678")

            # Create a mock legacy plan file
            legacy_file = storage_dir / f"{plan_id}.json"
            with legacy_file.open("w") as f:
                json.dump({"legacy": "data"}, f)

            # Create Portia instance with disk storage
            config = Config(
                storage_class=StorageClass.DISK,
                storage_path=str(storage_dir),
                planning_agent_type="default",
                execution_agent_type="default",
            )
            portia = Portia(config)

            assert portia._is_legacy_plan(plan_id) is True

    def test_is_legacy_plan_with_disk_storage_no_file(self) -> None:
        """Test _is_legacy_plan returns False when no file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            plan_id = PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678")

            # Create Portia instance with disk storage
            config = Config(
                storage_class=StorageClass.DISK,
                storage_path=str(storage_dir),
                planning_agent_type="default",
                execution_agent_type="default",
            )
            portia = Portia(config)

            assert portia._is_legacy_plan(plan_id) is False

    def test_is_legacy_plan_with_memory_storage(self) -> None:
        """Test _is_legacy_plan returns False with memory storage."""
        config = Config(
            storage_class=StorageClass.MEMORY,
            planning_agent_type="default",
            execution_agent_type="default",
        )
        portia = Portia(config)
        plan_id = PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678")

        assert portia._is_legacy_plan(plan_id) is False

    def test_load_plan_by_uuid_raises_legacy_deprecation_error(self) -> None:
        """Test that _load_plan_by_uuid raises LegacyPlanDeprecationError for legacy plans."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            plan_id = PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678")

            # Create a mock legacy plan file
            legacy_file = storage_dir / f"{plan_id}.json"
            with legacy_file.open("w") as f:
                json.dump({"legacy": "data"}, f)

            # Create Portia instance with disk storage
            config = Config(
                storage_class=StorageClass.DISK,
                storage_path=str(storage_dir),
                planning_agent_type="default",
                execution_agent_type="default",
            )
            portia = Portia(config)

            # Mock the storage to raise an exception (simulating plan not found in new format)
            with mock.patch.object(portia.storage, "get_plan", side_effect=Exception("Not found")):
                with pytest.raises(LegacyPlanDeprecationError) as exc_info:
                    portia._load_plan_by_uuid(plan_id)

                assert str(plan_id) in str(exc_info.value)

    async def test_aload_plan_by_uuid_raises_legacy_deprecation_error(self) -> None:
        """Test that _aload_plan_by_uuid raises LegacyPlanDeprecationError for legacy plans."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            plan_id = PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678")

            # Create a mock legacy plan file
            legacy_file = storage_dir / f"{plan_id}.json"
            with legacy_file.open("w") as f:
                json.dump({"legacy": "data"}, f)

            # Create Portia instance with disk storage
            config = Config(
                storage_class=StorageClass.DISK,
                storage_path=str(storage_dir),
                planning_agent_type="default",
                execution_agent_type="default",
            )
            portia = Portia(config)

            # Mock the storage to raise an exception (simulating plan not found in new format)
            with mock.patch.object(portia.storage, "aget_plan", side_effect=Exception("Not found")):
                with pytest.raises(LegacyPlanDeprecationError) as exc_info:
                    await portia._aload_plan_by_uuid(plan_id)

                assert str(plan_id) in str(exc_info.value)

    def test_load_plan_by_uuid_raises_plan_not_found_for_non_legacy(self) -> None:
        """Test that _load_plan_by_uuid raises PlanNotFoundError for non-legacy plans."""
        config = Config(
            storage_class=StorageClass.MEMORY,
            planning_agent_type="default",
            execution_agent_type="default",
        )
        portia = Portia(config)
        plan_id = PlanUUID.from_string("plan-12345678-1234-5678-1234-567812345678")

        # Mock the storage to raise an exception
        with mock.patch.object(portia.storage, "get_plan", side_effect=Exception("Not found")):
            with pytest.raises(PlanNotFoundError):
                portia._load_plan_by_uuid(plan_id)


class TestMigrationScript:
    """Test the migration script functionality."""

    def test_find_legacy_plan_files(self) -> None:
        """Test finding legacy plan files in a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)

            # Create some test files
            (storage_dir / "plan-12345678-1234-5678-1234-567812345678.json").touch()
            (storage_dir / "plan-87654321-4321-8765-4321-876543218765.json").touch()
            (storage_dir / "not-a-plan.json").touch()
            (storage_dir / "plan-without-json").touch()

            plan_files = find_legacy_plan_files(storage_dir)

            assert len(plan_files) == 2
            assert all(f.name.startswith("plan-") and f.name.endswith(".json") for f in plan_files)

    def test_find_legacy_plan_files_nonexistent_directory(self) -> None:
        """Test finding legacy plan files in a non-existent directory."""
        non_existent_dir = Path("/non/existent/directory")
        plan_files = find_legacy_plan_files(non_existent_dir)
        assert plan_files == []

    def test_load_legacy_plan_success(self) -> None:
        """Test successfully loading a legacy plan."""
        plan = Plan(
            plan_context=PlanContext(query="test query", tool_ids=[]),
            steps=[Step(task="test task", inputs=[], output="$output")],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(plan.model_dump(mode="json"), f)
            temp_file = Path(f.name)

        try:
            loaded_plan, error = load_legacy_plan(temp_file)
            assert loaded_plan is not None
            assert error is None
            assert loaded_plan.plan_context.query == "test query"
        finally:
            temp_file.unlink()

    def test_load_legacy_plan_failure(self) -> None:
        """Test handling failure when loading a legacy plan."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json {")
            temp_file = Path(f.name)

        try:
            loaded_plan, error = load_legacy_plan(temp_file)
            assert loaded_plan is None
            assert error is not None
            assert "json" in error.lower() or "expecting" in error.lower()
        finally:
            temp_file.unlink()

    def test_export_plan_to_json(self) -> None:
        """Test exporting a plan to JSON with metadata."""
        plan = Plan(
            plan_context=PlanContext(query="test query", tool_ids=[]),
            steps=[Step(task="test task", inputs=[], output="$output")],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            output_file = export_plan_to_json(plan, output_dir, "test_export.json")

            assert output_file.exists()

            with output_file.open("r") as f:
                exported_data = json.load(f)

            assert "migration_metadata" in exported_data
            assert "plan_data" in exported_data
            assert exported_data["migration_metadata"]["plan_id"] == str(plan.id)
            assert exported_data["migration_metadata"]["original_query"] == "test query"
            assert exported_data["plan_data"]["plan_context"]["query"] == "test query"

    def test_generate_migration_report(self) -> None:
        """Test generating a migration report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            failed_files = [(Path("test1.json"), "Error 1"), (Path("test2.json"), "Error 2")]

            report_file = generate_migration_report(5, 3, failed_files, output_dir)

            assert report_file.exists()

            with report_file.open("r") as f:
                report_data = json.load(f)

            assert report_data["migration_summary"]["total_files_found"] == 5
            assert report_data["migration_summary"]["successful_exports"] == 3
            assert report_data["migration_summary"]["failed_files"] == 2
            assert report_data["migration_summary"]["success_rate"] == "60.0%"
            assert len(report_data["failed_files"]) == 2

    @mock.patch("portia.migrate_legacy_plans.find_legacy_plan_files")
    @mock.patch("portia.migrate_legacy_plans.load_legacy_plan")
    @mock.patch("portia.migrate_legacy_plans.export_plan_to_json")
    @mock.patch("portia.migrate_legacy_plans.generate_migration_report")
    def test_main_dry_run(
        self,
        mock_generate_report: MagicMock,
        mock_export: MagicMock,
        mock_load: MagicMock,
        mock_find: MagicMock,
    ) -> None:
        """Test the main migration script in dry-run mode."""
        # Mock the plan object
        mock_plan = Plan(
            plan_context=PlanContext(query="test", tool_ids=[]),
            steps=[Step(task="test", inputs=[], output="$output")],
        )

        mock_find.return_value = [Path("test.json")]
        mock_load.return_value = (mock_plan, None)

        with mock.patch("sys.argv", ["migrate_legacy_plans.py", "--dry-run"]):
            result = migrate_main()

        assert result == 0
        mock_find.assert_called_once()
        mock_load.assert_called_once_with(Path("test.json"))
        mock_export.assert_not_called()  # Should not export in dry-run mode
        mock_generate_report.assert_not_called()  # Should not generate report in dry-run mode

    @mock.patch("portia.migrate_legacy_plans.find_legacy_plan_files")
    def test_main_no_files_found(self, mock_find: MagicMock) -> None:
        """Test the main migration script when no files are found."""
        mock_find.return_value = []

        with mock.patch("sys.argv", ["migrate_legacy_plans.py"]):
            result = migrate_main()

        assert result == 0
        mock_find.assert_called_once()

    @mock.patch("portia.migrate_legacy_plans.find_legacy_plan_files")
    @mock.patch("portia.migrate_legacy_plans.load_legacy_plan")
    def test_main_with_failed_files(self, mock_load: MagicMock, mock_find: MagicMock) -> None:
        """Test the main migration script with some failed files."""
        mock_find.return_value = [Path("test.json")]
        mock_load.return_value = (None, "Load error")

        with mock.patch("sys.argv", ["migrate_legacy_plans.py"]):
            result = migrate_main()

        assert result == 1  # Should return 1 when there are failures
        mock_find.assert_called_once()
        mock_load.assert_called_once()