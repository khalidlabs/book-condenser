# Changelog

## Unreleased

## 0.2.1

### Fixed

- Repair missing `supports` relations for essential major propositions in the analytical map, so synthesis validation no longer fails when the model omits support links.

## 0.2.0

### Changed

- Final selection allocates retained words per chapter in proportion to each included chapter's source length, replacing the flat `--chapter-max-share` ceiling.

### Removed

- `--chapter-max-share` CLI flag. Per-chapter budgets are now derived automatically from parsed structure and `--target-ratio`.

## 0.1.3

- New analytical framework for intellectual-structure analysis and passage selection.
- Remove bundled Walden sample output artifacts from the repository.

## 0.1.2

- Create a unique timestamped output subfolder for each abridgement run by default.
- Add `--reuse-output-dir` to overwrite a fixed output folder when desired.
- Expand README documentation and include a public-domain Walden EPUB example with sample output artifacts.

## 0.1.1

- Corrected package metadata links to point to `khalidlabs/book-condenser`.

## 0.1.0

- Initial public-release preparation.
- Added installable Python package metadata and `book-condenser` CLI entry point.
- Added PolyForm Noncommercial 1.0.0 license, security policy, contribution guide, and release documentation.
- Added focused unit tests, ruff linting, and GitHub Actions CI.
- Removed generated/private book artifacts from the release tree.
