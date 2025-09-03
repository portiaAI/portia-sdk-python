"""Tests for the loops module."""

from portia.builder.loops import (
    LoopBlock,
    LoopBlockType,
    LoopStepResult,
    LoopType,
)


class TestLoopBlockType:
    """Test the LoopBlockType enum."""

    def test_enum_values(self) -> None:
        """Test that LoopBlockType has the expected values."""
        assert LoopBlockType.START == "START"
        assert LoopBlockType.END == "END"

        # Test that we can access all values
        values = list(LoopBlockType)
        assert len(values) == 2
        assert LoopBlockType.START in values
        assert LoopBlockType.END in values


class TestLoopType:
    """Test the LoopType enum."""

    def test_enum_values(self) -> None:
        """Test that LoopType has the expected values."""
        assert LoopType.CONDITIONAL == "CONDITIONAL"
        assert LoopType.FOR_EACH == "FOR_EACH"

        # Test that we can access all values
        values = list(LoopType)
        assert len(values) == 2
        assert LoopType.CONDITIONAL in values
        assert LoopType.FOR_EACH in values


class TestLoopBlock:
    """Test the LoopBlock model."""

    def test_valid_loop_block_with_both_indexes(self) -> None:
        """Test creating a LoopBlock with both start and end indexes."""
        loop_block = LoopBlock(start_step_index=0, end_step_index=5)

        assert loop_block.start_step_index == 0
        assert loop_block.end_step_index == 5

    def test_valid_loop_block_with_only_start_index(self) -> None:
        """Test creating a LoopBlock with only start index (end can be None)."""
        loop_block = LoopBlock(start_step_index=0, end_step_index=None)

        assert loop_block.start_step_index == 0
        assert loop_block.end_step_index is None

    def test_valid_loop_block_with_large_indexes(self) -> None:
        """Test creating a LoopBlock with large index values."""
        loop_block = LoopBlock(start_step_index=100, end_step_index=200)

        assert loop_block.start_step_index == 100
        assert loop_block.end_step_index == 200

    def test_loop_block_with_zero_indexes(self) -> None:
        """Test creating a LoopBlock with zero indexes."""
        loop_block = LoopBlock(start_step_index=0, end_step_index=0)

        assert loop_block.start_step_index == 0
        assert loop_block.end_step_index == 0

    def test_loop_block_with_negative_start_index(self) -> None:
        """Test creating a LoopBlock with negative start index."""
        loop_block = LoopBlock(start_step_index=-1, end_step_index=5)

        assert loop_block.start_step_index == -1
        assert loop_block.end_step_index == 5

    def test_loop_block_with_negative_end_index(self) -> None:
        """Test creating a LoopBlock with negative end index."""
        loop_block = LoopBlock(start_step_index=0, end_step_index=-5)

        assert loop_block.start_step_index == 0
        assert loop_block.end_step_index == -5

    def test_loop_block_validation_start_required(self) -> None:
        """Test that start_step_index is required and cannot be None."""
        # Check that start_step_index field is required (no default value)
        start_field = LoopBlock.model_fields["start_step_index"]
        assert start_field.default is not None  # PydanticUndefined means it's required

        # Check that end_step_index field is also required (no default value)
        end_field = LoopBlock.model_fields["end_step_index"]
        assert end_field.default is not None  # PydanticUndefined means it's required

        # But end_step_index can accept None as a value due to Union[int, None] type
        assert end_field.annotation == int | type(None)

    def test_loop_block_validation_end_can_be_none(self) -> None:
        """Test that end_step_index can be None."""
        loop_block = LoopBlock(start_step_index=0, end_step_index=None)

        assert loop_block.start_step_index == 0
        assert loop_block.end_step_index is None

    def test_loop_block_field_descriptions(self) -> None:
        """Test that field descriptions are properly set."""
        # Get the model fields
        start_field = LoopBlock.model_fields["start_step_index"]
        end_field = LoopBlock.model_fields["end_step_index"]

        assert start_field.description == "The index of the first step in the loop."
        assert end_field.description == "The index of the last step in the loop."

    def test_loop_block_repr(self) -> None:
        """Test the string representation of LoopBlock."""
        loop_block = LoopBlock(start_step_index=1, end_step_index=10)
        repr_str = repr(loop_block)

        assert "LoopBlock" in repr_str
        assert "start_step_index=1" in repr_str
        assert "end_step_index=10" in repr_str

    def test_loop_block_equality(self) -> None:
        """Test LoopBlock equality comparison."""
        loop_block1 = LoopBlock(start_step_index=1, end_step_index=10)
        loop_block2 = LoopBlock(start_step_index=1, end_step_index=10)
        loop_block3 = LoopBlock(start_step_index=1, end_step_index=11)

        assert loop_block1 == loop_block2
        assert loop_block1 != loop_block3
        assert loop_block1 != "not a loop block"


class TestLoopStepResult:
    """Test the LoopStepResult model."""

    def test_valid_loop_step_result(self) -> None:
        """Test creating a valid LoopStepResult."""
        result = LoopStepResult(
            block_type=LoopBlockType.START,
            value="test_value",
            loop_result=True,
            start_index=0,
            end_index=5,
        )

        assert result.block_type == LoopBlockType.START
        assert result.value == "test_value"
        assert result.loop_result is True
        assert result.start_index == 0
        assert result.end_index == 5

    def test_loop_step_result_without_value(self) -> None:
        """Test creating a LoopStepResult without a value (should default to None)."""
        result = LoopStepResult(
            block_type=LoopBlockType.END, loop_result=False, start_index=10, end_index=15
        )

        assert result.block_type == LoopBlockType.END
        assert result.value is None
        assert result.loop_result is False
        assert result.start_index == 10
        assert result.end_index == 15

    def test_loop_step_result_with_none_value(self) -> None:
        """Test creating a LoopStepResult with explicit None value."""
        result = LoopStepResult(
            block_type=LoopBlockType.START,
            value=None,
            loop_result=True,
            start_index=5,
            end_index=10,
        )

        assert result.block_type == LoopBlockType.START
        assert result.value is None
        assert result.loop_result is True
        assert result.start_index == 5
        assert result.end_index == 10

    def test_loop_step_result_with_complex_value(self) -> None:
        """Test creating a LoopStepResult with a complex value."""
        complex_value = {"key": "value", "list": [1, 2, 3]}
        result = LoopStepResult(
            block_type=LoopBlockType.START,
            value=complex_value,
            loop_result=True,
            start_index=0,
            end_index=1,
        )

        assert result.block_type == LoopBlockType.START
        assert result.value == complex_value
        assert result.loop_result is True
        assert result.start_index == 0
        assert result.end_index == 1

    def test_loop_step_result_with_zero_indexes(self) -> None:
        """Test creating a LoopStepResult with zero indexes."""
        result = LoopStepResult(
            block_type=LoopBlockType.START, loop_result=False, start_index=0, end_index=0
        )

        assert result.start_index == 0
        assert result.end_index == 0

    def test_loop_step_result_with_negative_indexes(self) -> None:
        """Test creating a LoopStepResult with negative indexes."""
        result = LoopStepResult(
            block_type=LoopBlockType.END, loop_result=True, start_index=-5, end_index=-1
        )

        assert result.start_index == -5
        assert result.end_index == -1

    def test_loop_step_result_field_descriptions(self) -> None:
        """Test that field descriptions are properly set."""
        # Get the model fields
        value_field = LoopStepResult.model_fields["value"]

        assert value_field.description == "The value of the loop step."

    def test_loop_step_result_repr(self) -> None:
        """Test the string representation of LoopStepResult."""
        result = LoopStepResult(
            block_type=LoopBlockType.START,
            value="test",
            loop_result=True,
            start_index=1,
            end_index=10,
        )
        repr_str = repr(result)

        assert "LoopStepResult" in repr_str
        assert "block_type=<LoopBlockType.START: 'START'>" in repr_str
        assert "value='test'" in repr_str
        assert "loop_result=True" in repr_str
        assert "start_index=1" in repr_str
        assert "end_index=10" in repr_str

    def test_loop_step_result_equality(self) -> None:
        """Test LoopStepResult equality comparison."""
        result1 = LoopStepResult(
            block_type=LoopBlockType.START,
            value="test",
            loop_result=True,
            start_index=1,
            end_index=10,
        )
        result2 = LoopStepResult(
            block_type=LoopBlockType.START,
            value="test",
            loop_result=True,
            start_index=1,
            end_index=10,
        )
        result3 = LoopStepResult(
            block_type=LoopBlockType.END,
            value="test",
            loop_result=True,
            start_index=1,
            end_index=10,
        )

        assert result1 == result2
        assert result1 != result3
        assert result1 != "not a loop step result"

    def test_loop_step_result_with_different_block_types(self) -> None:
        """Test LoopStepResult with different block types."""
        start_result = LoopStepResult(
            block_type=LoopBlockType.START, loop_result=True, start_index=0, end_index=5
        )

        end_result = LoopStepResult(
            block_type=LoopBlockType.END, loop_result=False, start_index=0, end_index=5
        )

        assert start_result.block_type == LoopBlockType.START
        assert end_result.block_type == LoopBlockType.END
        assert start_result != end_result

    def test_loop_step_result_with_different_loop_results(self) -> None:
        """Test LoopStepResult with different loop result values."""
        true_result = LoopStepResult(
            block_type=LoopBlockType.START, loop_result=True, start_index=0, end_index=5
        )

        false_result = LoopStepResult(
            block_type=LoopBlockType.START, loop_result=False, start_index=0, end_index=5
        )

        assert true_result.loop_result is True
        assert false_result.loop_result is False
        assert true_result != false_result


class TestLoopModelsIntegration:
    """Test integration between different loop models."""

    def test_loop_block_with_loop_step_result_indexes(self) -> None:
        """Test that LoopBlock indexes can be used with LoopStepResult."""
        loop_block = LoopBlock(start_step_index=0, end_step_index=5)
        step_result = LoopStepResult(
            block_type=LoopBlockType.START,
            loop_result=True,
            start_index=loop_block.start_step_index,
            end_index=loop_block.end_step_index or 0,
        )

        assert step_result.start_index == loop_block.start_step_index
        assert step_result.end_index == loop_block.end_step_index

    def test_loop_type_with_loop_block_type(self) -> None:
        """Test that LoopType and LoopBlockType work together."""
        # This test ensures the enums are compatible and can be used together
        start_block = LoopBlockType.START
        end_block = LoopBlockType.END
        conditional_loop = LoopType.CONDITIONAL
        for_each_loop = LoopType.FOR_EACH

        # Verify all enum values are strings
        assert isinstance(start_block, str)
        assert isinstance(end_block, str)
        assert isinstance(conditional_loop, str)
        assert isinstance(for_each_loop, str)

        # Verify they have different values
        assert start_block != end_block
        assert conditional_loop != for_each_loop

    def test_model_serialization(self) -> None:
        """Test that loop models can be serialized and deserialized."""
        loop_block = LoopBlock(start_step_index=1, end_step_index=10)
        step_result = LoopStepResult(
            block_type=LoopBlockType.START,
            value="test_value",
            loop_result=True,
            start_index=1,
            end_index=10,
        )

        # Test serialization to dict
        loop_block_dict = loop_block.model_dump()
        step_result_dict = step_result.model_dump()

        assert isinstance(loop_block_dict, dict)
        assert isinstance(step_result_dict, dict)
        assert loop_block_dict["start_step_index"] == 1
        assert step_result_dict["block_type"] == "START"

        # Test deserialization from dict
        new_loop_block = LoopBlock.model_validate(loop_block_dict)
        new_step_result = LoopStepResult.model_validate(step_result_dict)

        assert new_loop_block == loop_block
        assert new_step_result == step_result
