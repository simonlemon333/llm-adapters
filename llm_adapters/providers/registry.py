"""Provider registry — maps provider names to adapter classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_adapters.config import ProviderConfig, get_provider_config
from llm_adapters.exceptions import ProviderNotFoundError

if TYPE_CHECKING:
    from llm_adapters.providers.base import BaseLLMProvider


# Built-in provider factories (lazy imports to avoid loading all adapters upfront)
_BUILTIN_PROVIDERS: dict[str, str] = {
    "openai": "llm_adapters.providers.openai.OpenAIProvider",
    "anthropic": "llm_adapters.providers.anthropic.AnthropicProvider",
    "deepseek": "llm_adapters.providers.deepseek.DeepSeekProvider",
    "ollama": "llm_adapters.providers.ollama.OllamaProvider",
}


class ProviderRegistry:
    """Registry that creates and caches provider instances."""

    def __init__(self) -> None:
        self._instances: dict[str, BaseLLMProvider] = {}
        self._custom: dict[str, type[BaseLLMProvider]] = {}

    def register(self, name: str, cls: type[BaseLLMProvider]) -> None:
        """Register a custom provider class."""
        self._custom[name] = cls

    def get(self, name: str, config: ProviderConfig | None = None) -> BaseLLMProvider:
        """Get or create a provider instance by name."""
        if name in self._instances:
            return self._instances[name]

        cfg = config or get_provider_config(name)

        if name in self._custom:
            instance = self._custom[name](cfg)
        elif name in _BUILTIN_PROVIDERS:
            instance = self._import_and_create(name, cfg)
        else:
            raise ProviderNotFoundError(
                f"Unknown provider: {name!r}. "
                f"Available: {sorted(set(_BUILTIN_PROVIDERS) | set(self._custom))}",
                provider=name,
            )

        self._instances[name] = instance
        return instance

    def _import_and_create(self, name: str, config: ProviderConfig) -> BaseLLMProvider:
        """Lazily import a built-in provider class and instantiate it."""
        import importlib

        dotted = _BUILTIN_PROVIDERS[name]
        module_path, class_name = dotted.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(config)

    async def close_all(self) -> None:
        """Close all cached provider instances."""
        for provider in self._instances.values():
            await provider.close()
        self._instances.clear()
