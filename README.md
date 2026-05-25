# Book Condenser

**Create a shorter reading edition of a nonfiction book while preserving the author's original prose and line of reasoning.**

Book Condenser converts an EPUB, PDF, DOCX, TXT, or Markdown book into a professionally formatted condensed edition. Instead of asking a model to rewrite the book as a summary, it identifies the original passages needed to preserve the book's central question, thesis, essential concepts, arguments, evidence, qualifications, and conclusion. Those passages are then assembled verbatim into a shorter reading document.

Where omission would create a difficult jump, Book Condenser can insert a brief, clearly labelled **editorial transition**. The substantive content remains the author's text; generated transitions serve only as disclosed continuity aids.

> Read less of the book, without replacing the book with a summary.

## Why this approach

A conventional summary tells the reader what a book concludes. A condensed reading edition should retain enough of the original work for the reader to follow how the author arrives there.

Book Condenser is designed for nonfiction works in which the development of the argument matters. It attempts to retain:

- the problem the author is addressing;
- the governing thesis or explanatory position;
- essential terms in the author's own usage;
- major propositions and the reasoning or evidence that supports them;
- substantive limitations, objections, or qualifications;
- the final conclusion or implications.


## How it works

1. **Parse and validate the source**  
   The program extracts readable text, recovers chapter structure, identifies back matter, and writes a local structure report. If the source cannot be recovered reliably, processing stops before model calls are made.

2. **Build an analytical map**  
   The model first performs a structural reading of the book, then analyzes individual chapters. It synthesizes a map of the central question, thesis, essential terms, major propositions, supporting argument or evidence, qualifications, and conclusion.

3. **Select original passages**  
   Candidate passages are nominated as contiguous blocks from the source text and tagged against the analytical map. The program protects the passages required to preserve essential propositions together with their support.

4. **Control length and redundancy**  
   The selected passages are balanced under the target word budget, while avoiding excessive repetition or dominance by a single chapter.

5. **Check analytical completeness**  
   A final review checks whether a reader can still reconstruct the author's problem, terms, argument, supporting material, qualifications, and conclusion.

6. **Bridge difficult omissions, when enabled**  
   Short editorial transitions may be generated between noncontiguous retained passages. They are separately validated, strictly limited in length, and visibly labelled as non-original text.

7. **Render the reading edition**  
   The resulting Markdown, PDF, and optional DOCX documents contain retained original passages, omission markers, and any approved editorial transitions.

## Key properties

- **Quotation-dominant output.** Retained substantive passages are retrieved verbatim from the source.
- **Analytical selection.** Passage selection is organized around the author's question, thesis, terms, propositions, support, and conclusion rather than only chapter coverage.
- **Argument preservation.** An essential proposition is not retained without mapped supporting reasoning or evidence.
- **Disclosed transitions.** Generated bridges are italicized and labelled `Editorial transition`; they are never presented as the author's prose.
- **Separate word accounting.** The target ratio applies to retained source words. Editorial-transition words are reported separately.
- **Traceability.** The run produces structure, analytical-map, selection, quality-control, and transition-validation artifacts.

## Supported inputs

| Format | Guidance |
|---|---|
| EPUB | Preferred when available. EPUB generally provides the cleanest reading order and chapter structure. |
| PDF | Text-based PDFs are supported. Scanned or image-only PDFs require OCR first. |
| DOCX | Supported when headings and paragraphs are reasonably structured. |
| TXT / Markdown | Supported; headings improve section recovery. |

Book Condenser detects and excludes common non-reading material such as references, bibliography, notes, acknowledgments, index entries, and publisher matter from passage selection and source-word budgeting.

## Requirements

- Python 3.10 or newer.
- An OpenAI API key for full condensation runs.
- A source book that you are permitted to process and store.

Use Book Condenser with public-domain books, your own writing, or works for which you have appropriate permission. A condensed edition may still contain substantial copyrighted source text.

## Installation

Install from PyPI:

```bash
pip install book-condenser
```

Install from a local checkout:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Set the API key for full runs:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

The default model is `gpt-5-mini`. It may be changed with either an environment variable or a command-line option:

```bash
export OPENAI_MODEL="gpt-5-mini"
```

or:

```bash
book-condenser book.epub --model gpt-5-mini
```

## Quick start

### 1. Validate the source locally

Begin with `--parse-only`. This checks text recovery and section structure without sending book content to the API.

```bash
book-condenser path/to/book.epub \
  --output-dir out \
  --parse-only
```

Review:

```text
parsed_structure_report.md
```

Proceed when the report identifies the chapters correctly, excludes back matter appropriately, and indicates that model-based extraction may proceed.

### 2. Produce a condensed reading edition

```bash
book-condenser path/to/book.epub \
  --output-dir out \
  --target-ratio 0.25 \
  --apply-qc \
  --transitions minimal \
  --pdf-font-size 14
```

A `--target-ratio` of `0.25` aims to retain approximately 25% of the source words as verbatim reading passages. Any approved editorial transitions are counted and reported separately.

The primary output is:

```text
reading_abridgement.pdf
```

## Editorial transitions

Extracting passages can produce abrupt changes in chronology or argument. Book Condenser can generate short bridges that orient the reader without substituting generated prose for the omitted material.

Transitions appear in the edition as:

```markdown
*Editorial transition: The omitted discussion develops further examples of this mechanism before the author turns to its practical consequence.*
```

They are:

- generated only after the final retained passages have been selected;
- checked in a separate validation pass;
- limited in length and total share of the reading edition;
- explicitly identified as editorial rather than original text.

### Transition controls

| Option | Meaning | Default |
|---|---|---:|
| `--transitions` | `none`, `minimal`, or `guided` reader-visible transition policy | `minimal` |
| `--max-transition-words` | Maximum words in one approved transition | `45` |
| `--max-transition-share` | Maximum transition words as a proportion of retained source words | `0.02` |
| `--transition-batch-size` | Candidate discontinuities processed per transition call | `12` |

Examples:

```bash
# Strictly extractive output with omission markers only
book-condenser book.epub --transitions none --apply-qc

# More orientation for a book with substantial narrative or argumentative jumps
book-condenser book.epub --transitions guided --apply-qc
```

## PDF reading edition

The PDF output is designed for comfortable reading on a small tablet:

- 7 × 10 inch portrait pages by default;
- 14 pt serif body text by default;
- generous line spacing and clean chapter openings;
- visible omission markers between retained passages;
- italicized editorial transitions, where approved;
- restrained running headers and page numbers.

| Option | Meaning | Default |
|---|---|---:|
| `--pdf-page-size` | `small-tablet`, `a5`, or `large-tablet` | `small-tablet` |
| `--pdf-font-size` | Body font size from 11 to 20 pt | `14` |
| `--pdf-font` | `auto`, `georgia`, `dejavu serif`, `dejavuserif`, or `times` | `auto` |
| `--no-docx` | Do not generate the optional DOCX copy | off |

Example for larger type:

```bash
book-condenser book.epub \
  --output-dir out \
  --pdf-font-size 15 \
  --no-docx \
  --apply-qc
```

## Source format guidance

### EPUB

EPUB is the preferred input because it generally contains explicit reading order and section metadata. Book Condenser supports:

- EPUB 2 `toc.ncx` navigation;
- EPUB 3 navigation documents;
- anchored subsections;
- visible-heading recovery when navigation data are incomplete;
- common imperfect EPUB structures.

### PDF

Text-based PDFs are supported. The program uses embedded bookmarks where available and can attempt section recovery from visible headings.

For scanned or image-only files, perform OCR before using Book Condenser. When a text-based PDF has unreliable chapter detection, supply a manual chapter map:

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

Pages are 1-indexed. Back matter remains visible in the structure report but is excluded from passage selection and source-word budgeting.

Final retained words are allocated across included chapters in proportion to each chapter's source length (with a 600-word floor per chapter and a 15% cap boost for chapters marked high priority in the structural overview). Mandatory analytical coverage and continuity anchors may exceed a chapter's cap when required.

## Main controls

| Argument | Meaning | Default |
|---|---|---:|
| `--target-ratio` | Final retained source-word ratio | `0.25` |
| `--candidate-ratio` | Candidate quotation share per chapter before global pruning | `0.42` |
| `--coverage-mode` | Continuity coverage after mandatory analytical coverage: `all`, `major`, or `none` | `all` |
| `--emphasis` | Additional selection guidance supplied to the model | built-in analytical emphasis |
| `--chapter-chunk-words` | Split unusually long chapters above this word count for model calls | `18000` |
| `--max-structural-words` | Maximum opening and ending words supplied to the inspectional overview | `24000` |
| `--score-batch-size` | Candidate blocks scored per call | `20` |
| `--apply-qc` | Apply final add/remove quality-control recommendations within budget tolerance | off |
| `--chapter-map` | Manual chapter/page map for difficult PDFs | none |
| `--parse-only` | Parse and validate without model calls | off |
| `--model` | OpenAI model supporting structured outputs | `OPENAI_MODEL` or `gpt-5-mini` |
| `--retries` | Maximum retries per API call | `3` |
| `--output-dir` | Parent directory for generated run folders | `abridgement_output` |
| `--reuse-output-dir` | Write directly into the specified output folder and replace prior generated artifacts there | off |
| `--verbose` | Enable detailed logging | off |

Transition and PDF controls are listed in their respective sections above.

## Outputs

A full run creates a timestamped folder by default:

```text
out/book-<timestamp>/
    reading_abridgement.pdf
    reading_abridgement.md
    reading_abridgement.docx          # unless --no-docx
    analytical_reading_guide.md
    selection_audit.md
    parsed_structure_report.md
    book_metadata.json
    book_paragraphs.jsonl
    structural_overview.json
    chapter_analysis/
    chapter_analyses.json
    analytical_map.json
    chapter_candidates/
    scored_candidates.json
    global_selection.json
    quality_control.json
    editorial_transitions.json
    editorial_transition_validation.json
```

### Reader-facing files

| File | Purpose |
|---|---|
| `reading_abridgement.pdf` | Final tablet-friendly condensed reading edition |
| `reading_abridgement.docx` | Optional editable version of the edition |
| `analytical_reading_guide.md` | Map of the book's problem, unity, requirements, and retained reading path |

### Verification and audit files

| File | Purpose |
|---|---|
| `parsed_structure_report.md` | Verifies recovered sections and parser confidence before model processing |
| `selection_audit.md` | Reports retained-source ratio, analytical coverage, passage decisions, and approved transitions |
| `analytical_map.json` | Structured map of terms, propositions, support, qualifications, and conclusion |
| `quality_control.json` | Final analytical completeness and coherence review |
| `editorial_transition_validation.json` | Validation record for generated continuity bridges |

Keep output folders private by default. They may contain verbatim source passages, local file paths, and model-generated analytical metadata.

## Cost and privacy

`--parse-only` runs locally and does not require an API key or send book text to a model.

A full run sends structural context and source excerpts to the selected OpenAI model. API usage depends on source length, chapter structure, candidate-passage ratio, final quality-control use, and transition settings.

Do not process confidential, restricted, or copyrighted material unless your rights and provider settings permit that use.

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

The package exposes both:

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

Book Condenser is provided as-is and does not provide legal advice. You are responsible for ensuring that source material, model/API use, generated transitions, and generated outputs comply with applicable copyright law, contractual terms, platform policies, and other obligations.
