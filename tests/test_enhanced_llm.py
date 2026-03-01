"""Tests for the enhanced LLM wrapper."""

import pytest
from unittest.mock import MagicMock
from src.enhanced_llm import EnhancedChatModel, create_enhanced_llm, ENHANCEMENT_STYLES


class TestEnhancedChatModel:

    def test_string_prompt_gets_prefix(self):
        """String prompts should have the prefix prepended."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="test response")

        enhanced = EnhancedChatModel(mock_llm, "PREFIX INSTRUCTIONS")
        enhanced.invoke("Analyze this stock")

        call_args = mock_llm.invoke.call_args[0][0]
        assert call_args.startswith("PREFIX INSTRUCTIONS")
        assert "Analyze this stock" in call_args

    def test_list_prompt_gets_system_message(self):
        """Message list prompts should get a prepended system message."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="test response")

        enhanced = EnhancedChatModel(mock_llm, "PREFIX INSTRUCTIONS")
        enhanced.invoke([
            {"role": "system", "content": "You are a trader"},
            {"role": "user", "content": "What should I do?"},
        ])

        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) == 3  # prefix + system + user
        assert "PREFIX INSTRUCTIONS" in call_args[0].content

    def test_bind_tools_passes_through(self):
        """bind_tools should NOT be enhanced — passes through to base LLM."""
        mock_llm = MagicMock()
        enhanced = EnhancedChatModel(mock_llm, "PREFIX")
        enhanced.bind_tools(["tool1", "tool2"])
        mock_llm.bind_tools.assert_called_once_with(["tool1", "tool2"])

    def test_create_enhanced_llm_valid_style(self):
        """Factory function with valid style should work."""
        mock_llm = MagicMock()
        enhanced = create_enhanced_llm(mock_llm, style="financial_analysis")
        assert isinstance(enhanced, EnhancedChatModel)

    def test_create_enhanced_llm_invalid_style(self):
        """Factory function with invalid style should raise."""
        mock_llm = MagicMock()
        with pytest.raises(ValueError, match="Unknown style"):
            create_enhanced_llm(mock_llm, style="nonexistent")

    def test_all_styles_exist(self):
        """All documented styles should be in ENHANCEMENT_STYLES."""
        assert "financial_analysis" in ENHANCEMENT_STYLES
        assert "structured" in ENHANCEMENT_STYLES
        assert "few_shot" in ENHANCEMENT_STYLES

    def test_prefix_contains_data_requirements(self):
        """All prefixes should mention data citation requirements."""
        for style, prefix in ENHANCEMENT_STYLES.items():
            assert "10" in prefix or "data point" in prefix.lower(), (
                f"Style '{style}' doesn't mention data citation count"
            )

    def test_unknown_input_type_passthrough(self):
        """Unknown input types should pass through to base LLM."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="response")
        enhanced = EnhancedChatModel(mock_llm, "PREFIX")

        enhanced.invoke(42)
        mock_llm.invoke.assert_called_once_with(42)

    def test_getattr_proxies_to_base(self):
        """Attribute access should proxy to the base LLM."""
        mock_llm = MagicMock()
        mock_llm.some_attribute = "test_value"
        enhanced = EnhancedChatModel(mock_llm, "PREFIX")
        assert enhanced.some_attribute == "test_value"

    def test_string_prompt_prefix_separator(self):
        """Prefix and original prompt should be separated by double newline."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="response")
        enhanced = EnhancedChatModel(mock_llm, "MY PREFIX")
        enhanced.invoke("MY PROMPT")

        call_args = mock_llm.invoke.call_args[0][0]
        assert "MY PREFIX\n\nMY PROMPT" == call_args
