# Book Condenser

**Create a shorter reading edition of a nonfiction book from the author’s original passages.**

Book Condenser transforms an EPUB, PDF, DOCX, TXT, or Markdown book into a clean, tablet-friendly PDF abridgement. An AI model identifies the passages that carry the book’s central argument, evidence, concepts, chronology, and conclusions. The program then retrieves those passages from the source and assembles them into a shorter reading edition.

The result is **shorter than the source, richer than a summary, and faithful to the author’s voice**.

> **Preserve the author. Remove the excess.**

## How It Works

1. **Recover structure**  
   The program identifies chapters, reading order, and back matter, while cleaning common extraction artifacts.

2. **Validate the source**  
   A local structure report checks whether the recovered text is reliable before model-based selection begins.

3. **Select essential passages**  
   The model determines the nonfiction form and selects coherent original passages that carry the book’s intellectual or narrative arc.

4. **Balance the abridgement**  
   The program reduces redundancy, protects broad chapter coverage, limits overrepresentation of individual sections, and meets the requested target length.

5. **Produce the reading edition**  
   The retained source passages are rendered as a professionally formatted, large-type PDF for tablet reading.

The AI acts as an **editorial selector**. The final edition remains grounded in the author’s original text.

## Features

- Supports **EPUB, PDF, DOCX, TXT, and Markdown** input.
- Recovers structure from EPUB 2, EPUB 3, and text-based PDFs, including imperfect source files.
- Detects and excludes notes, bibliography, acknowledgments, indexes, and other non-reading matter.
- Stops before API calls when the parsed structure is unreliable or the source is likely image-only.
- Adapts passage selection to argumentative, historical, technical, biographical, and mixed nonfiction.
- Produces a **tablet-optimized PDF** as the primary output.
- Generates parsing and selection reports for traceability.

## Requirements

You need:

- Python 3.10 or newer.
- An OpenAI API key for full condensation runs.
- A source book you are legally allowed to process and store.

Use Book Condenser with public-domain works, your own material, or works for which you have appropriate permission. Generated editions contain substantial source text.

**EPUB is preferred** when available because it usually provides cleaner chapter structure and text than PDF.

## Installation

From PyPI, once released:

```bash
pip install book-condenser
```

From a local checkout:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

Optionally choose the model:

```bash
export OPENAI_MODEL="gpt-5-mini"
```

When `OPENAI_MODEL` is not set, the CLI uses `gpt-5-mini`.

## Quick Start

### 1. Check the source structure locally

Start with `--parse-only`. It validates the input and creates a report **without sending book text to the API**.

```bash
book-condenser path/to/book.epub \
  --output-dir out \
  --parse-only
```

Open the generated:

```text
parsed_structure_report.md
```

Proceed when chapters are detected correctly, back matter is excluded appropriately, and the report indicates that extraction may proceed.

### 2. Generate the condensed reading edition

```bash
book-condenser path/to/book.epub \
  --output-dir out \
  --target-ratio 0.25 \
  --coverage-mode all \
  --chapter-max-share 0.08 \
  --pdf-font-size 14 \
  --apply-qc
```

A target ratio of `0.25` aims to retain approximately one quarter of the original book’s words.

The primary output is:

```text
reading_abridgement.pdf
```

## PDF Reading Edition

The default PDF is designed for comfortable reading on a small tablet:

- 7 × 10 inch portrait pages.
- Large 14 pt serif body text.
- Generous line spacing and clean chapter openings.
- Discreet markers between separated retained passages.
- Restrained running headers and page numbers.

Useful controls:

| Option | Purpose | Default |
|---|---|---:|
| `--pdf-page-size` | `small-tablet`, `a5`, or `large-tablet` | `small-tablet` |
| `--pdf-font-size` | Body font size from 11 to 20 pt | `14` |
| `--pdf-font` | `auto`, `georgia`, `dejavu serif`, or `times` | `auto` |
| `--no-docx` | Skip optional DOCX output | off |

For larger text on a small screen:

```bash
book-condenser path/to/book.epub \
  --output-dir out \
  --pdf-font-size 15 \
  --no-docx \
  --apply-qc
```

## Source Format Guidance

### EPUB

EPUB is the recommended input. Book Condenser supports:

- EPUB 2 `toc.ncx` navigation.
- EPUB 3 navigation documents.
- Visible-heading recovery when navigation metadata is missing.
- Anchored subsections and common imperfect EPUB structures.

### PDF

Text-based PDFs are supported. The program uses bookmarks when available and can attempt to recover sections from visible headings.

For a scanned or image-only PDF, run OCR first. When chapter boundaries are unreliable, provide a manual chapter map.

```bash
book-condenser path/to/book.pdf \
  --chapter-map examples/chapter_map.json \
  --output-dir out \
  --parse-only
```

Example chapter map:

```json
[
  {"title": "Introduction", "start_page": 1, "end_page": 8},
  {"title": "Chapter One", "start_page": 9},
  {"title": "Chapter Two", "start_page": 28},
  {"title": "Bibliography", "start_page": 410}
]
```

Back matter remains visible in the structure report but is excluded from passage selection and word budgeting.

## Main Controls

| Argument | Meaning | Default |
|---|---|---:|
| `--target-ratio` | Approximate share of source words retained | `0.25` |
| `--candidate-ratio` | Candidate passage pool before global pruning | `0.42` |
| `--coverage-mode` | Section coverage rule: `all`, `major`, or `none` | `all` |
| `--chapter-max-share` | Nominal maximum share from one chapter | `0.08` |
| `--parse-only` | Validate parsing without API calls | off |
| `--apply-qc` | Apply final model-based quality review | off |
| `--chapter-map` | Manual page map for difficult PDFs | none |
| `--output-dir` | Parent directory for generated run folders | `abridgement_output` |
| `--reuse-output-dir` | Replace prior generated artifacts in that folder | off |

## Outputs

A full run creates a folder such as:

```text
out/book-<timestamp>/
    reading_abridgement.pdf
    parsed_structure_report.md
    selection_audit.md
    reading_abridgement.md
    reading_abridgement.docx
    book_metadata.json
    book_paragraphs.jsonl
    structural_overview.json
    chapter_candidates/
    scored_candidates.json
    global_selection.json
    quality_control.json
```

Files most users need:

| File | Purpose |
|---|---|
| `reading_abridgement.pdf` | Final tablet-friendly reading edition |
| `parsed_structure_report.md` | Verification that the source was parsed correctly |
| `selection_audit.md` | Record of coverage and passage-selection decisions |
| `reading_abridgement.docx` | Optional editable copy |

Keep output folders private by default. They may contain verbatim passages, local paths, and model-generated selection analysis.

## Cost and Privacy

`--parse-only` runs locally and does not require API calls.

A full run sends structural context and source excerpts to the configured OpenAI model. API usage increases with book length, candidate-pool size, and use of final quality-control review.

Do not process confidential or restricted material unless your rights and API/provider settings permit it.

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run checks:

```bash
ruff check .
pytest
python -m build
twine check dist/*
```

The package exposes:

```bash
book-condenser
```

and:

```bash
python -m book_condenser
```

## License

Book Condenser is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use requires a separate commercial license from the licensor.

## Disclaimer

Book Condenser is provided as-is and does not provide legal advice. You are responsible for ensuring that source material, API use, and generated outputs comply with applicable copyright law, contract terms, platform policies, and other obligations.
