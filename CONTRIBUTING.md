# Contributing

Thanks for helping improve Book Condenser.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the local checks before opening a pull request:

```bash
ruff check .
pytest
python -m build
twine check dist/*
```

## Test Data

Do not commit copyrighted books, generated abridgements, or full parsed source text. Use synthetic fixtures or public-domain material that is clearly safe to redistribute.

## Style

Keep behavior-preserving refactors separate from functional changes when possible. The CLI is the public interface, so changes to flags, defaults, output names, or file formats should include tests and README updates.

