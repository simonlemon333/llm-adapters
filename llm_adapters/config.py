"""Provider configuration and model mapping."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ProviderConfig:
    api_key: str = ""
    base_url: str = ""
    timeout: float = 60.0
    max_retries: int = 2
    extra: dict[str, str] = field(default_factory=dict)


# Default base URLs per provider
DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "deepseek": "https://api.deepseek.com/v1",
    "ollama": "http://localhost:11434",
}

# Environment variable names for API keys
API_KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def get_provider_config(provider_name: str, **overrides: str | float | int) -> ProviderConfig:
    """Build a ProviderConfig from env vars and defaults, with optional overrides."""
    env_var = API_KEY_ENV_VARS.get(provider_name, "")
    api_key = str(overrides.get("api_key", os.environ.get(env_var, "")))
    base_url = str(overrides.get("base_url", DEFAULT_BASE_URLS.get(provider_name, "")))
    timeout = float(overrides.get("timeout", 60.0))
    max_retries = int(overrides.get("max_retries", 2))
    return ProviderConfig(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
