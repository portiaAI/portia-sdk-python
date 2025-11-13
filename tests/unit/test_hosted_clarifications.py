"""Tests for hosted clarifications functionality."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from portia.clarification import HostedClarification, InputClarification
from portia.cloud import PortiaCloudClient
from portia.config import Config
from portia.prefixed_uuid import PlanRunUUID


class TestHostedClarifications:
    """Test suite for hosted clarifications."""

    def test_hosted_clarification_model(self):
        """Test that HostedClarification can be instantiated."""
        hosted_clarification = HostedClarification(
            url="https://example.com/clarifications/test-id",
            clarification_id="hclar-test123",
        )
        assert hosted_clarification.url == "https://example.com/clarifications/test-id"
        assert hosted_clarification.clarification_id == "hclar-test123"

    def test_hosted_clarification_serialization(self):
        """Test that HostedClarification serializes correctly."""
        hosted_clarification = HostedClarification(
            url="https://example.com/clarifications/test-id",
            clarification_id="hclar-test123",
        )
        data = hosted_clarification.model_dump()
        assert data["url"] == "https://example.com/clarifications/test-id"
        assert data["clarification_id"] == "hclar-test123"

    @patch("portia.cloud.httpx.Client")
    def test_create_hosted_clarification_api_call(self, mock_client_class):
        """Test that create_hosted_clarification makes correct API call."""
        # Setup
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "hclar-test123",
            "url": "https://example.com/clarifications/test-id",
            "status": "PENDING",
            "clarification_data": {
                "id": "clar_test",
                "category": "Input",
                "user_guidance": "Test guidance",
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        config = Config.from_default(
            portia_api_key="test-key",
            portia_api_endpoint="https://api.test.com",
            openai_api_key="test-openai-key",
        )

        # Create clarification
        clarification = InputClarification(
            argument_name="test_arg",
            user_guidance="Test guidance",
            plan_run_id=PlanRunUUID(),
        )

        plan_run_id = PlanRunUUID()
        client = PortiaCloudClient(config)

        # Execute
        result = client.create_hosted_clarification(
            plan_run_id=plan_run_id,
            clarification=clarification,
        )

        # Verify
        assert isinstance(result, HostedClarification)
        assert result.clarification_id == "hclar-test123"
        assert result.url == "https://example.com/clarifications/test-id"

        # Verify API call was made correctly
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert f"/api/v0/plan_runs/{plan_run_id}/hosted_clarifications/" in call_args[0][0]
        assert "clarification_data" in call_args[1]["json"]

    @patch("portia.execution_agents.base_execution_agent.PortiaCloudClient")
    def test_base_agent_add_clarification_with_hosted_enabled(self, mock_cloud_client_class):
        """Test that add_clarification creates hosted clarification when enabled."""
        from portia.config import Config
        from portia.end_user import EndUser
        from portia.execution_agents.one_shot_agent import OneShotAgent
        from portia.plan import Plan
        from portia.plan_run import PlanRun

        # Setup mock
        mock_hosted_clarification = HostedClarification(
            url="https://example.com/clarifications/test-id",
            clarification_id="hclar-test123",
        )
        mock_client = MagicMock()
        mock_client.create_hosted_clarification.return_value = mock_hosted_clarification
        mock_cloud_client_class.return_value = mock_client

        # Create test objects
        config = Config.from_default(
            use_hosted_clarifications=True,
            openai_api_key="test-key",
        )

        plan = Plan(
            query="Test query",
            steps=[{"task": "Test task", "tool_name": "test_tool", "output": "$test"}],
        )
        plan_run = PlanRun(plan=plan)
        end_user = EndUser(external_id="test-user")

        # Create mock agent memory
        mock_agent_memory = MagicMock()

        # Create agent
        agent = OneShotAgent(
            plan=plan,
            plan_run=plan_run,
            config=config,
            end_user=end_user,
            agent_memory=mock_agent_memory,
        )

        # Create and add clarification
        clarification = InputClarification(
            argument_name="test_arg",
            user_guidance="Test guidance",
            plan_run_id=plan_run.id,
        )

        agent.add_clarification(clarification)

        # Verify hosted clarification was created
        assert len(agent.new_clarifications) == 1
        assert isinstance(agent.new_clarifications[0], HostedClarification)
        assert agent.new_clarifications[0].url == "https://example.com/clarifications/test-id"

        # Verify API was called
        mock_client.create_hosted_clarification.assert_called_once_with(
            plan_run_id=plan_run.id,
            clarification=clarification,
        )

    def test_base_agent_add_clarification_without_hosted_enabled(self):
        """Test that add_clarification adds normal clarification when hosted is disabled."""
        from portia.config import Config
        from portia.end_user import EndUser
        from portia.execution_agents.one_shot_agent import OneShotAgent
        from portia.plan import Plan
        from portia.plan_run import PlanRun

        # Create test objects
        config = Config.from_default(
            use_hosted_clarifications=False,
            openai_api_key="test-key",
        )

        plan = Plan(
            query="Test query",
            steps=[{"task": "Test task", "tool_name": "test_tool", "output": "$test"}],
        )
        plan_run = PlanRun(plan=plan)
        end_user = EndUser(external_id="test-user")

        # Create mock agent memory
        mock_agent_memory = MagicMock()

        # Create agent
        agent = OneShotAgent(
            plan=plan,
            plan_run=plan_run,
            config=config,
            end_user=end_user,
            agent_memory=mock_agent_memory,
        )

        # Create and add clarification
        clarification = InputClarification(
            argument_name="test_arg",
            user_guidance="Test guidance",
            plan_run_id=plan_run.id,
        )

        agent.add_clarification(clarification)

        # Verify normal clarification was added
        assert len(agent.new_clarifications) == 1
        assert isinstance(agent.new_clarifications[0], InputClarification)
        assert agent.new_clarifications[0].argument_name == "test_arg"
