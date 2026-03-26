# llm-adapters

A minimal, auditable Python library for calling multiple LLM providers.
Built as a lightweight alternative to LiteLLM after the [March 2026 PyPI supply chain attack](https://blog.pypi.org/posts/2026-03-24-litellm-incident/).

## Why?

- **Only 2 dependencies**: `httpx` + `pydantic` — no official SDKs, no bloat
- **Each provider is a single file** — fully readable and auditable
- **All deps pinned with hash verification** in CI
- **< 2000 lines of code** — anyone can review the entire codebase in minutes

## Install

```bash
pip install llm-dial
```

## Quick Start

```python
from llm_adapters import LLMClient

client = LLMClient()

# Unified interface — auto-routes to the correct provider
response = await client.chat(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.content)

# Streaming
async for chunk in client.chat_stream(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Tell me a story"}],
):
    print(chunk.content, end="")
```

## Supported Providers

| Provider | Models | Auth |
|----------|--------|------|
| **OpenAI** | GPT-4o, GPT-4, o1, o3, etc. | `OPENAI_API_KEY` |
| **Anthropic** | Claude 4, Claude 3.5, etc. | `ANTHROPIC_API_KEY` |
| **DeepSeek** | deepseek-chat, deepseek-coder | `DEEPSEEK_API_KEY` |
| **Ollama** | Any local model (llama3, mistral, etc.) | None (local) |

## Three Ways to Use

```
Your app
    │
    ├── As a library: from llm_adapters import LLMClient
    │
    ├── As a proxy: app → localhost:4000 → llm-adapters serve → LLM APIs
    │       (drop-in LiteLLM sidecar replacement, zero code changes)
    │
    └── As a gateway base: app → llm-gateway → llm-adapters → LLM APIs
            (add routing / auth / rate limiting / budget / observability)
```

## Comparison with LiteLLM

| | LiteLLM | llm-adapters |
|---|---------|-------------|
| Dependencies | 100+ transitive | 2 (httpx + pydantic) |
| Install size | ~50MB | <1MB |
| Supply chain risk | Compromised March 2026 | Minimal attack surface |
| Supported models | 100+ | Core 4 providers (enough for most) |
| Code size | Tens of thousands of lines | <2000 lines |
| Auditability | Complex | One file per adapter |

## Security

This library was born from the LiteLLM supply chain incident. We take dependency security seriously:

- Minimal dependency tree (2 direct deps)
- All versions pinned with SHA-256 hashes in CI
- Automated `pip-audit` on every PR
- SBOM included in releases
- Published via GitHub Actions OIDC (Trusted Publishers) — no PyPI tokens

## License

MIT
