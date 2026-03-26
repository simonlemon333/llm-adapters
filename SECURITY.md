# Security Policy

## Why This Project Exists

This library was created after the [LiteLLM PyPI supply chain attack](https://blog.pypi.org/posts/2026-03-24-litellm-incident/) (March 2026), where malicious versions stole SSH keys, cloud credentials, and Kubernetes secrets via `.pth` file injection.

We take supply chain security seriously.

## Security Measures

- **Minimal dependencies**: Only `httpx` + `pydantic` — smallest possible attack surface
- **Trusted Publishers**: Published via GitHub Actions OIDC — no PyPI tokens exist
- **Anti-.pth scanning**: CI automatically rejects packages containing `.pth` files
- **Dependency auditing**: `pip-audit` runs on every PR via GitHub Actions
- **Dependabot**: Automatic PRs for vulnerable dependencies
- **Code auditability**: <2000 lines of code, each provider is a single file

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT open a public issue**
2. Email: simonlemon333@gmail.com
3. Include: description, reproduction steps, potential impact
4. You will receive a response within 48 hours

## Verifying Package Integrity

Every release includes SHA-256 hashes. To verify:

```bash
pip download llm-adapters --no-deps --no-cache-dir
sha256sum llm_adapters-*.whl
# Compare with hash published in GitHub Release notes
```

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Current |
