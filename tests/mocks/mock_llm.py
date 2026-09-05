"""Backward-compatible re-export of the packaged mock provider (HH-09).

``MockLLMProvider`` moved into the package (``ctxai.agent.llm.mock_provider``)
so the HH-09 mock-provider benchmark mode works from any installed ctxai,
including clean installations without the test suite. Every historical
``tests.mocks.mock_llm`` import keeps working through this module.
"""

from ctxai.agent.llm.mock_provider import (  # noqa: F401
    MockLLMProvider,
    create_mock_response,
)

__all__ = ["MockLLMProvider", "create_mock_response"]
