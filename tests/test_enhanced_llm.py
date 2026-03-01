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
        assert "execution_params_only" in ENHANCEMENT_STYLES

    def test_prefix_contains_data_requirements(self):
        """Data-citation styles should mention data citation requirements.
        execution_params_only is exempt — it's a lightweight structural-only style."""
        data_citation_styles = {
            k: v for k, v in ENHANCEMENT_STYLES.items()
            if k != "execution_params_only"
        }
        for style, prefix in data_citation_styles.items():
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

    def test_execution_parameters_in_financial_prefix(self):
        """FINANCIAL_ANALYSIS_PREFIX must contain ## EXECUTION PARAMETERS block."""
        from src.enhanced_llm import FINANCIAL_ANALYSIS_PREFIX
        assert "## EXECUTION PARAMETERS" in FINANCIAL_ANALYSIS_PREFIX
        assert "Stop-Loss:" in FINANCIAL_ANALYSIS_PREFIX
        assert "Price Target:" in FINANCIAL_ANALYSIS_PREFIX
        assert "Position Size:" in FINANCIAL_ANALYSIS_PREFIX

    def test_execution_parameters_in_structured_prefix(self):
        """STRUCTURED_OUTPUT_PREFIX must contain ## EXECUTION PARAMETERS block."""
        from src.enhanced_llm import STRUCTURED_OUTPUT_PREFIX
        assert "## EXECUTION PARAMETERS" in STRUCTURED_OUTPUT_PREFIX
        assert "Stop-Loss:" in STRUCTURED_OUTPUT_PREFIX
        assert "Price Target:" in STRUCTURED_OUTPUT_PREFIX
        assert "Position Size:" in STRUCTURED_OUTPUT_PREFIX

    def test_execution_parameters_in_few_shot_prefix(self):
        """FEW_SHOT_PREFIX must contain ## EXECUTION PARAMETERS block."""
        from src.enhanced_llm import FEW_SHOT_PREFIX
        assert "## EXECUTION PARAMETERS" in FEW_SHOT_PREFIX
        assert "Stop-Loss:" in FEW_SHOT_PREFIX
        assert "Price Target:" in FEW_SHOT_PREFIX
        assert "Position Size:" in FEW_SHOT_PREFIX

    def test_execution_params_only_style_exists(self):
        """execution_params_only style should exist and only contain the EXECUTION PARAMETERS block."""
        from src.enhanced_llm import EXECUTION_PARAMS_ONLY_PREFIX, ENHANCEMENT_STYLES
        assert "execution_params_only" in ENHANCEMENT_STYLES
        assert "## EXECUTION PARAMETERS" in EXECUTION_PARAMS_ONLY_PREFIX
        assert "Stop-Loss:" in EXECUTION_PARAMS_ONLY_PREFIX
        # Should NOT contain the data citation requirements (that's for local models)
        assert "cite at least 10" not in EXECUTION_PARAMS_ONLY_PREFIX
