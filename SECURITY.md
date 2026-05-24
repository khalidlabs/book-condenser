# Security Policy

## Reporting Issues

Please report security issues privately to the project maintainer before opening a public issue. Include the affected version, a clear reproduction path, and any relevant logs with secrets removed.

## Secrets

Book Condenser reads API credentials from environment variables such as `OPENAI_API_KEY`. Do not commit `.env` files, shell history, generated logs, or output artifacts containing credentials.

If an API key is exposed, revoke it with the provider immediately, create a replacement key, and remove the exposed value from the repository and any published history.

## Generated Artifacts

Generated files may contain substantial verbatim text from source books and local filesystem paths. Treat `out/`, `books/`, and similar working directories as private unless you have verified that every file is safe to publish.

