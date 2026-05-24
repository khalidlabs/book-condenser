# Book Condenser

## Read the Essential Book

Book Condenser creates an extractive abridgement of a nonfiction book. An AI model identifies the original passages that carry the book's central argument, evidence, concepts, turning points, and conclusions. The software then assembles those passages verbatim into a shorter, beautifully formatted reading edition.

This approach preserves what makes a serious book valuable: the author's reasoning, voice, and choice of evidence. Many nonfiction books develop their core ideas through repetition, extended examples, and supporting detail. By retaining the passages that do the essential intellectual work, Book Condenser makes the book more efficient to read while keeping the reader in direct contact with the original text.

The result is a condensed, tablet-friendly PDF designed for focused reading: shorter than the source, richer than a summary, and faithful to the author.

This tool is intended for books you own the rights to process, public-domain works, or other material you are legally allowed to transform and store. Generated outputs may contain substantial verbatim source text.

## Features

- Supports EPUB, PDF, DOCX, TXT, and Markdown input.
- Validates parsing with `--parse-only` before making API calls.
- Preserves chronology and argument structure through subtype-aware selection rules.
- Protects broad coverage with `--coverage-mode all` and per-section concentration limits.
- Produces `reading_abridgement.pdf` as the primary reader-facing output.
- Writes audit artifacts so users can inspect selected passages, scores, coverage, and quality-control decisions.

## Installation

From PyPI after release:

```bash
pip install book-condenser
```

For local development from a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Set your OpenAI API key in the environment before running the full pipeline:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

You can also set `OPENAI_MODEL`; otherwise the CLI defaults to `gpt-5-mini`.

## Quick Start

Validate parsing before any API calls:

```bash
book-condenser path/to/public-domain-book.epub \
  --output-dir out/example \
  --parse-only
```

Review `out/example/parsed_structure_report.md`. Continue only if chapter and back-matter detection look plausible.

Generate a reading edition:

```bash
book-condenser path/to/public-domain-book.epub \
  --output-dir out/example \
  --target-ratio 0.25 \
  --coverage-mode all \
  --chapter-max-share 0.08 \
  --apply-qc
```

For PDFs with unreliable bookmarks, provide a manual chapter map:

```bash
book-condenser path/to/public-domain-book.pdf \
  --chapter-map examples/chapter_map.json \
  --output-dir out/example \
  --parse-only
```

The root `book_condenser.py` file is a compatibility launcher. Prefer the installed `book-condenser` command for normal use.

## Key Controls

| Argument | Purpose | Default |
|---|---|---:|
| `--target-ratio` | Target proportion of source words retained | `0.25` |
| `--candidate-ratio` | Candidate pool before global pruning | `0.42` |
| `--coverage-mode` | Section coverage rule: `all`, `major`, or `none` | `all` |
| `--chapter-max-share` | Maximum nominal share of final text from one chapter | `0.08` |
| `--chapter-map` | Manual PDF section/page map when bookmarks are unreliable | none |
| `--parse-only` | Validate structure and cleanup without API calls | off |
| `--apply-qc` | Apply final model review within constraints | off |
| `--pdf-page-size` | `small-tablet`, `a5`, or `large-tablet` | `small-tablet` |
| `--pdf-font-size` | Body type size between 11 and 20 pt | `14.0` |
| `--pdf-font` | `auto`, `georgia`, `dejavu serif`, or `times` | `auto` |
| `--no-docx` | Skip optional DOCX output | off |

## Outputs

```text
out/example/
    parsed_structure_report.md
    book_metadata.json
    book_paragraphs.jsonl
    structural_overview.json
    chapter_candidates/
    scored_candidates.json
    global_selection.json
    quality_control.json
    selection_audit.md
    reading_abridgement.md
    reading_abridgement.pdf
    reading_abridgement.docx
```

`reading_abridgement.pdf` is the primary reading edition. `selection_audit.md` records subtype classification, chapter balance, selected passage functions, scores, protected anchors, and locations.

Treat the entire output directory as private by default. It can contain verbatim source text, local paths, and model-generated analysis.

## Manual Chapter Map Format

Pages are 1-indexed. `end_page` is optional; when omitted, the next section's `start_page - 1` is used.

```json
[
  {"title": "Prologue", "start_page": 1, "end_page": 8},
  {"title": "Chapter One", "start_page": 9},
  {"title": "Chapter Two", "start_page": 28},
  {"title": "Bibliography", "start_page": 410}
]
```

Back matter headings are retained in the parse audit but excluded from selection and source-word budgeting.

## Source Format Guidance

Prefer EPUB when available. PDFs may require a manual chapter map and inspection of the parse-only report. If a PDF is scanned or image-only, run OCR first.

The parser supports EPUB 2 `toc.ncx`, EPUB 3 navigation documents, semantic back-matter signals, anchored subsections, PDF bookmarks, visible-heading fallback, and common PDF text cleanup.

## Cost and Privacy

Full runs send selected source excerpts and structural context to the configured OpenAI model. Use `--parse-only` to inspect local parsing before any API calls. Larger books, higher `--candidate-ratio`, and `--apply-qc` increase token usage and cost.

Do not process confidential, copyrighted, or sensitive books unless your API/provider settings and legal rights allow that use.

## Development

Run checks locally:

```bash
ruff check .
pytest
python -m build
twine check dist/*
```

The package exposes `book-condenser` as a console script and `python -m book_condenser` as a module entry point.

## Release Checklist

1. Confirm the repository root is this project directory, not a parent home directory.
2. Verify no `.env`, `books/`, `out/`, generated abridgements, or copyrighted fixtures are tracked.
3. Run `ruff check .`, `pytest`, `python -m build`, and `twine check dist/*`.
4. Configure PyPI trusted publishing for `khalidlabs/book-condenser` using the `Publish to PyPI` workflow.
5. Publish a GitHub release or run the publish workflow manually after package install and CLI smoke tests pass.

## License

Book Condenser is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use is not permitted by this license without a separate commercial license from the licensor.

## Disclaimer

Book Condenser is provided as-is and does not provide legal advice. You are responsible for ensuring that your source material and generated outputs comply with copyright law, contract terms, platform policies, and any other obligations that apply to your use.
