"""
LLM Provider Factory.

Creates LLM providers based on configuration.
"""

import os
from typing import Optional

from rich.console import Console

from ..config import AgentLLMConfig
from .base import BaseLLMProvider

console = Console()


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create_provider(config: AgentLLMConfig) -> BaseLLMProvider:
        """Create one provider or an explicitly enabled fallback chain."""
        primary = LLMProviderFactory._create_single(config)
        if not config.fallback_enabled:
            return primary

        from .fallback import FallbackProvider

        providers = [(config.provider.lower(), primary)]
        for name in config.fallback_providers:
            if name.lower() == config.provider.lower():
                continue
            fallback_config = AgentLLMConfig(
                provider=name,
                api_key=config.get_api_key_for_provider(name),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
            )
            providers.append((name.lower(), LLMProviderFactory._create_single(fallback_config)))
        return FallbackProvider(
            providers,
            allow_boundary_crossing=config.allow_fallback_boundary_crossing,
        )

    @staticmethod
    def _create_single(config: AgentLLMConfig) -> BaseLLMProvider:
        """
        Create LLM provider based on configuration.

        Args:
            config: LLM configuration

        Returns:
            Initialized LLM provider

        Raises:
            ValueError: If provider is not supported
        """
        provider = config.provider.lower()

        if provider == "openrouter":
            from .openrouter_provider import OpenRouterProvider
            return OpenRouterProvider(config)

        elif provider == "ollama":
            from .ollama_provider import OllamaProvider
            return OllamaProvider(config)

        elif provider == "anthropic":
            from .anthropic_provider import AnthropicProvider
            return AnthropicProvider(config)

        elif provider == "openai":
            from .openai_provider import OpenAIProvider
            return OpenAIProvider(config)

        elif provider == "github-copilot":
            from .github_copilot_provider import GitHubCopilotProvider
            return GitHubCopilotProvider(config)

        elif provider == "custom":
            from .custom_provider import CustomProvider
            return CustomProvider(config)

        elif provider == "nvidia":
            from .custom_provider import CustomProvider
            return CustomProvider(config)

        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: openrouter, ollama, anthropic, openai, github-copilot, custom, nvidia"
            )

    @staticmethod
    def create_from_name(
        provider_name: str,
        model_name: str | None = None,
        **kwargs,
    ) -> BaseLLMProvider:
        """
        Create provider from name and model.

        Args:
            provider_name: Provider name (openrouter, ollama, etc.)
            model_name: Model name
            **kwargs: Additional config parameters

        Returns:
            Initialized provider
        """
        config = AgentLLMConfig(
            provider=provider_name,
            model=model_name,
            **kwargs,
        )
        return LLMProviderFactory.create_provider(config)

    @staticmethod
    def get_recommended_models() -> dict:
        """
        Get recommended model configurations.

        Returns:
            Dict of recommended configs by use case
        """
        return {
            "best_coding": {
                "provider": "openrouter",
                "model": "anthropic/claude-3.5-sonnet",
                "description": "Best overall for coding tasks",
                "cost": "$$",
            },
            "best_reasoning": {
                "provider": "openrouter",
                "model": "openai/o1-mini",
                "description": "Best for complex reasoning and planning",
                "cost": "$$$$",
            },
            "fast_cheap": {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "description": "Fast and affordable",
                "cost": "$",
            },
            "free_default": {
                "provider": "openrouter",
                "model": "deepseek/deepseek-r1:free",
                "description": "Free model with reasoning (default)",
                "cost": "Free",
            },
            "very_cheap": {
                "provider": "openrouter",
                "model": "deepseek/deepseek-chat",
                "description": "Very cheap, decent quality",
                "cost": "¢",
            },
            "local_best": {
                "provider": "ollama",
                "model": "codellama:13b",
                "description": "Best free local model for coding",
                "cost": "Free",
            },
            "local_fast": {
                "provider": "ollama",
                "model": "qwen2.5-coder:7b",
                "description": "Fast local model",
                "cost": "Free",
            },
        }

    @staticmethod
    def get_architect_editor_pair(preset: str = "default") -> tuple:
        """
        Get recommended architect/editor pair.

        Args:
            preset: Preset name (default, budget, local, mixed)

        Returns:
            Tuple of (architect_config, editor_config)
        """
        presets = {
            "default": {
                "architect": ("openrouter", "openai/o1-mini"),
                "editor": ("openrouter", "anthropic/claude-3.5-sonnet"),
                "description": "Best quality + cost balance",
                "cost": "$$",
            },
            "premium": {
                "architect": ("openrouter", "openai/o1"),
                "editor": ("openrouter", "anthropic/claude-3-opus"),
                "description": "Best quality, high cost",
                "cost": "$$$$$",
            },
            "budget": {
                "architect": ("openrouter", "openai/gpt-4o"),
                "editor": ("openrouter", "openai/gpt-4o-mini"),
                "description": "Good quality, lower cost",
                "cost": "$",
            },
            "cheap": {
                "architect": ("openrouter", "deepseek/deepseek-r1"),
                "editor": ("openrouter", "deepseek/deepseek-chat"),
                "description": "Cheapest cloud option",
                "cost": "¢",
            },
            "local": {
                "architect": ("ollama", "codellama:34b"),
                "editor": ("ollama", "codellama:13b"),
                "description": "Fully local (free), needs good hardware",
                "cost": "Free",
            },
            "mixed": {
                "architect": ("openrouter", "openai/o1-mini"),
                "editor": ("ollama", "codellama:13b"),
                "description": "Cloud planning + local implementation",
                "cost": "$",
            },
        }

        if preset not in presets:
            console.print(f"[yellow]Unknown preset: {preset}, using 'default'[/yellow]")
            preset = "default"

        config = presets[preset]
        arch_provider, arch_model = config["architect"]
        edit_provider, edit_model = config["editor"]

        architect_config = AgentLLMConfig(
            provider=arch_provider,
            model=arch_model,
            temperature=0.3,  # Lower for planning
        )

        editor_config = AgentLLMConfig(
            provider=edit_provider,
            model=edit_model,
            temperature=0.7,  # Higher for implementation
        )

        console.print(f"\n[cyan]Using preset: {preset}[/cyan]")
        console.print(f"  Description: {config['description']}")
        console.print(f"  Cost: {config['cost']}")
        console.print(f"  Architect: {arch_model}")
        console.print(f"  Editor: {edit_model}\n")

        return architect_config, editor_config

    @staticmethod
    def check_provider_availability(provider: str) -> tuple[bool, str]:
        """
        Check if a provider is available and configured.

        Args:
            provider: Provider name

        Returns:
            Tuple of (is_available, message)
        """
        # Check keystore for stored API keys
        try:
            from ...auth.keystore import get_keystore
            keystore = get_keystore()
        except ImportError:
            keystore = None

        if provider == "openrouter":
            # Check environment variable first, then keystore
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key and keystore:
                api_key = keystore.get_key("openrouter")

            if not api_key:
                return False, "Not logged in (run: ctxai login openrouter)"
            return True, "OpenRouter configured"

        elif provider == "ollama":
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    return True, "Ollama running"
                return False, "Ollama not responding"
            except Exception:
                return False, "Ollama not running (start with: ollama serve)"

        elif provider == "anthropic":
            # Check environment variable first, then keystore
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key and keystore:
                api_key = keystore.get_key("anthropic")

            if not api_key:
                return False, "ANTHROPIC_API_KEY not set"
            return True, "Anthropic configured"

        elif provider == "openai":
            # Check environment variable first, then keystore
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key and keystore:
                api_key = keystore.get_key("openai")

            if not api_key:
                return False, "OPENAI_API_KEY not set"
            return True, "OpenAI configured"

        elif provider == "github-copilot":
            # Check environment variable first, then keystore
            token = os.getenv("GITHUB_COPILOT_TOKEN")
            if not token and keystore:
                token = keystore.get_key("github-copilot")

            if not token:
                return False, "Not logged in (run: ctxai login github-copilot)"
            return True, "GitHub Copilot configured"

        elif provider == "custom":
            # Custom provider requires base_url and api_key
            # Check environment first, then config file
            api_key = os.getenv("CUSTOM_API_KEY") or os.getenv("MODAL_API_KEY")
            base_url = os.getenv("CUSTOM_BASE_URL") or os.getenv("MODAL_BASE_URL")

            # Also check config file if not provided
            if not api_key or not base_url:
                try:
                    from ...config import ConfigManager
                    cm = ConfigManager()
                    cfg = cm.load()
                    pconfig = cfg.get_provider_config("custom")
                    if not api_key and pconfig.api_key:
                        api_key = pconfig.api_key
                    if not base_url and pconfig.base_url:
                        base_url = pconfig.base_url
                except Exception:
                    pass

            if not api_key:
                return False, "CUSTOM_API_KEY/MODAL_API_KEY not set (or set providers.custom.api_key in config)"
            if not base_url:
                return False, "CUSTOM_BASE_URL/MODAL_BASE_URL not set (or set providers.custom.base_url in config)"
            return True, "Custom provider configured"

        elif provider == "nvidia":
            # NVIDIA NIM provider requires base_url and api_key
            api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVAPI_KEY")
            base_url = os.getenv("NVIDIA_BASE_URL")

            # Also check config file if not provided
            if not api_key:
                try:
                    from ...config import ConfigManager
                    cm = ConfigManager()
                    cfg = cm.load()
                    pconfig = cfg.get_provider_config("nvidia")
                    if not api_key and pconfig.api_key:
                        api_key = pconfig.api_key
                except Exception:
                    pass

            if not api_key:
                return False, "NVIDIA_API_KEY/NVAPI_KEY not set (or set providers.nvidia.api_key in config)"
            return True, "NVIDIA provider configured"

        return False, f"Unknown provider: {provider}"

    @staticmethod
    def print_provider_status():
        """Print status of all providers."""
        console.print("\n[bold cyan]Provider Status:[/bold cyan]\n")

        providers = ["openrouter", "github-copilot", "ollama", "anthropic", "openai", "nvidia", "custom"]

        for provider in providers:
            available, message = LLMProviderFactory.check_provider_availability(provider)
            status = "[OK]" if available else "[X]"
            color = "green" if available else "red"

            # Format provider name
            provider_name = provider.replace("-", " ").title()

            console.print(f"  [{color}]{status}[/{color}] {provider_name}: {message}")

        console.print()

    @staticmethod
    def get_setup_instructions() -> str:
        """Get setup instructions for providers."""
        return """
Setup Instructions:

1. OpenRouter (recommended):
   Option A - OAuth (easiest):
     • Run: ctxai login openrouter
     • One-click authentication in browser
     • API key stored securely

   Option B - Manual:
     • Get API key: https://openrouter.ai/keys
     • Set: export OPENROUTER_API_KEY=your-key-here

   • Access to 100+ models (Claude, GPT-4o, o1, DeepSeek, etc.)

2. GitHub Copilot (if you have subscription):
   • Run: ctxai login github-copilot
   • Follow device code flow
   • Visit github.com/login/device
   • Access to GPT-4, Claude, and more via Copilot

3. Ollama (local, free):
   • Install: https://ollama.ai
   • Start: ollama serve
   • Pull model: ollama pull codellama:13b
   • No API key needed, runs locally

4. Anthropic (direct):
   • Get API key: https://console.anthropic.com/
   • Set: export ANTHROPIC_API_KEY=your-key-here
   • Direct access to Claude models

5. OpenAI (direct):
   • Get API key: https://platform.openai.com/api-keys
   • Set: export OPENAI_API_KEY=your-key-here
   • Direct access to GPT models

6. Custom (OpenAI-compatible endpoints like Modal):
    • Get API key from your custom endpoint provider
    • Set: export CUSTOM_API_KEY=your-key-here (or MODAL_API_KEY for Modal)
    • Configure in code:
      from ctxai.agent.config import AgentLLMConfig
      config = AgentLLMConfig(
          provider="custom",
          model="your-model-name",
          api_key="your-api-key",
          base_url="https://api.us-west-2.modal.direct/v1"
      )

7. NVIDIA NIM (NVIDIA NIM endpoints):
    • Get API key: https://build.nvidia.com/nim
    • Set: export NVIDIA_API_KEY=your-key-here (or NVAPI_KEY)
    • Configure in code or config:
      from ctxai.agent.config import AgentLLMConfig
      config = AgentLLMConfig(
          provider="nvidia",
          model="nvidia/your-model",
          api_key="your-api-key",
          base_url="https://integrate.api.nvidia.com/v1"
      )
"""
