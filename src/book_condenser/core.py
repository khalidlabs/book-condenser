#!/usr/bin/env python3
"""
book_condenser.py

Create a full extractive abridgement of a rights-cleared/public-domain nonfiction book.
The LLM selects paragraph identifiers and writes short analytical metadata only.
All quotation text in the final abridgement is retrieved verbatim from the source.

Supported input formats:
    .epub, .pdf, .docx, .txt, .md

Outputs:
    <output_dir>/
        book_metadata.json
        book_paragraphs.jsonl
        structural_overview.json
        chapter_candidates/*.json
        scored_candidates.json
        global_selection.json
        quality_control.json
        reading_abridgement.md
        reading_abridgement.pdf
        reading_abridgement.docx       (unless --no-docx)

Example:
    export OPENAI_API_KEY="..."
    book-condenser book.epub \
        --output-dir out/project_maven \
        --target-ratio 0.25 \
        --emphasis "Central thesis, major claims, evidence, concepts, mechanisms, counterarguments, and conclusions."

Design:
    1. Parse the source and assign stable paragraph IDs.
    2. Read the TOC-equivalent chapter list plus introduction and ending material.
    3. Ask the LLM for a structural overview.
    4. Ask the LLM to nominate contiguous quotation blocks from each chapter.
    5. Ask the LLM to score candidate blocks; Python selects blocks under budget.
    6. Ask the LLM for a final coherence/redundancy review.
    7. Assemble exact source quotations into Markdown, PDF, and DOCX.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import posixpath
import re
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger("nonfiction_extractive_condenser")
T = TypeVar("T", bound=BaseModel)

WORD_RE = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
HEADING_RE = re.compile(
    r"^\s*(chapter\s+([0-9ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\b.*|\d+\s+[A-Z][^\n]{2,100}|summary of the framework.*|"
    r"introduction\b.*|prologue\b.*|preface\b.*|foreword\b.*|"
    r"epilogue\b.*|conclusion\b.*|afterword\b.*|"
    r"part\s+([0-9ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\b.*|"
    r"appendix\b.*|notes\b.*|bibliography\b.*|references\b.*|acknowledg(e)?ments\b.*|index\b.*)$",
    flags=re.IGNORECASE,
)

FRONT_PATTERNS = ("introduction", "prologue", "preface", "foreword")
END_PATTERNS = ("epilogue", "conclusion", "afterword")
EXCLUDE_PATTERNS = (
    "copyright", "contents", "table of contents", "bibliography", "references", "index",
    "notes", "acknowledg", "list of interviews", "written correspondence", "interviews and written",
    "photo insert", "photo-insert", "illustrations", "plates", "image credits", "about the author"
)


# ---------------------------------------------------------------------------
# Local source models
# ---------------------------------------------------------------------------

@dataclass
class Paragraph:
    paragraph_id: str
    chapter_id: str
    chapter_title: str
    index: int
    text: str
    word_count: int
    page: int | None = None


@dataclass
class Chapter:
    chapter_id: str
    title: str
    paragraph_ids: list[str]
    word_count: int
    kind: str = "body"


@dataclass
class Book:
    title: str
    source_path: str
    chapters: list[Chapter]
    paragraphs: dict[str, Paragraph]
    total_words: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectedBlock:
    block_id: str
    chapter_id: str
    chapter_title: str
    paragraph_ids: list[str]
    word_count: int
    selection_reason: str
    importance: str
    themes: list[str]
    redundancy_risk: str
    block_function: str = "supporting"
    text: str = ""
    score: float = 0.0
    redundant_with: list[str] = field(default_factory=list)
    protected_anchor: bool = False


# ---------------------------------------------------------------------------
# Structured LLM response models
# ---------------------------------------------------------------------------

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CoreClaim(StrictModel):
    claim: str
    role_in_argument: str
    importance: str


class KeyConcept(StrictModel):
    name: str
    function_in_argument: str
    importance: str


class EvidenceOrCase(StrictModel):
    description: str
    evidentiary_role: str
    importance: str


class ChapterPriority(StrictModel):
    chapter_id: str
    priority: str
    selection_focus: str
    suggested_budget_fraction: float


class StructuralOverview(StrictModel):
    nonfiction_form: Literal[
        "argumentative",
        "historical_investigative_narrative",
        "biography_memoir",
        "technical_explanatory",
        "case_based_policy_business",
        "mixed",
    ]
    central_question: str
    governing_thesis: str
    overview: str
    chronological_or_logical_arc: list[str]
    core_claims: list[CoreClaim]
    key_concepts: list[KeyConcept]
    evidence_and_cases: list[EvidenceOrCase]
    counterarguments_or_limitations: list[str]
    chapter_priorities: list[ChapterPriority]
    selection_rules: list[str]


class CandidateBlockResponse(StrictModel):
    block_id: str
    paragraph_ids: list[str]
    block_function: Literal[
        "setup", "definition", "claim", "mechanism", "evidence", "representative_episode",
        "turning_point", "consequence", "counterargument", "interpretation", "conclusion"
    ]
    selection_reason: str
    importance: str
    themes: list[str]
    redundancy_risk: str


class ChapterCandidateResponse(StrictModel):
    chapter_id: str
    chapter_contribution: str
    candidate_blocks: list[CandidateBlockResponse]
    omitted_material_description: list[str]


class ScoredBlockResponse(StrictModel):
    block_id: str
    chronological_or_logical_necessity: int = Field(ge=0, le=5)
    institutional_causal_or_argumentative_importance: int = Field(ge=0, le=5)
    turning_point_or_conclusion_value: int = Field(ge=0, le=5)
    explanatory_density: int = Field(ge=0, le=5)
    readability: int = Field(ge=0, le=3)
    redundancy_penalty: int = Field(ge=0, le=5)
    excessive_detail_penalty: int = Field(ge=0, le=5)
    context_dependence_penalty: int = Field(ge=0, le=3)
    redundant_with: list[str]
    rationale: str


class ScoringResponse(StrictModel):
    scores: list[ScoredBlockResponse]


class TransitionNote(StrictModel):
    before_block_id: str
    note: str


class QualityControlResponse(StrictModel):
    assessment: str
    missing_coverage: list[str]
    remove_block_ids: list[str]
    add_block_ids: list[str]
    transition_notes: list[TransitionNote]


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def normalize_text(text: str) -> str:
    """Normalize extraction artefacts without rewriting the author's substantive prose."""
    text = text.replace("\u00ad", "").replace("\xa0", " ")
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    # PDF extraction often represents a broken hyphen with a replacement/noncharacter glyph.
    text = re.sub(r"(?<=\w)[\ufffd\ufffe](?=\w)", "-", text)
    text = text.replace("\ufffd", "").replace("\ufffe", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    # Repair specific drop-cap/OCR splits observed at paragraph openings without
    # collapsing legitimate prose such as "A condensed..." or "I think...".
    text = re.sub(r"^T\s+he\b", "The", text)
    text = re.sub(r"^D\s+uring\b", "During", text)
    text = re.sub(r"\ba(?=[A-Z][a-z]{3,}\b)", "a ", text)
    text = re.sub(r"(?<=[A-Z&])(?=Dr\.)", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def paragraph_split(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n+", text)
    cleaned: list[str] = []
    for block in blocks:
        block = normalize_text(re.sub(r"\n+", " ", block))
        if block and word_count(block) >= 2:
            cleaned.append(block)
    return cleaned


def slug_id(prefix: str, index: int) -> str:
    return f"{prefix}{index:03d}"


def canonical_title(title: str) -> str:
    title = normalize_text(title).lower()
    title = re.sub(r"^[\W_]+|[\W_]+$", "", title)
    title = re.sub(r"\s+", " ", title)
    return title


def classify_chapter(title: str) -> str:
    """Classify reader-visible section titles only; internal filenames are not headings."""
    lower = canonical_title(title)
    excluded_patterns = (
        r"^(copyright|contents|table of contents)\b",
        r"^(bibliography|references|works cited)\b",
        r"^(index)\b",
        r"^(notes|endnotes)\b",
        r"^(acknowledg(e)?ments?)\b",
        r"^(list of interviews|written correspondence|interviews and written)\b",
        r"^(photo[\s-]?insert|illustrations|plates|image credits)\b",
        r"^(about the author)\b",
        r"^(unmapped front matter)\b",
    )
    if any(re.match(pattern, lower) for pattern in excluded_patterns):
        return "exclude"
    if any(re.match(rf"^{re.escape(p)}\b", lower) for p in FRONT_PATTERNS):
        return "front"
    if any(re.match(rf"^{re.escape(p)}\b", lower) for p in END_PATTERNS):
        return "ending"
    return "body"


def safe_json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        payload = data.model_dump()
    else:
        payload = data
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def model_dump_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [model_dump_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: model_dump_jsonable(v) for k, v in obj.items()}
    return obj


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    front = max_words * 2 // 3
    back = max_words - front
    return " ".join(words[:front]) + "\n\n[... omitted for input-length control ...]\n\n" + " ".join(words[-back:])


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def build_book_from_sections(
    source_path: Path,
    title: str,
    sections: list[tuple[str, list[tuple[str, int | None]]]],
    diagnostics: dict[str, Any] | None = None,
) -> Book:
    paragraphs: dict[str, Paragraph] = {}
    chapters: list[Chapter] = []

    usable_sections = [(t.strip() or f"Section {i + 1}", ps) for i, (t, ps) in enumerate(sections) if ps]
    if not usable_sections:
        raise ValueError("No readable paragraph text could be extracted from the input file.")

    for chapter_no, (chapter_title, blocks) in enumerate(usable_sections, start=1):
        chapter_id = slug_id("CH", chapter_no)
        para_ids: list[str] = []
        chapter_words = 0
        for para_no, (text, page) in enumerate(blocks, start=1):
            text = normalize_text(text)
            if not text or word_count(text) < 2:
                continue
            pid = f"{chapter_id}-P{para_no:04d}"
            wc = word_count(text)
            paragraphs[pid] = Paragraph(
                paragraph_id=pid,
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                index=para_no,
                text=text,
                word_count=wc,
                page=page,
            )
            para_ids.append(pid)
            chapter_words += wc
        if para_ids:
            chapter_kind = classify_chapter(chapter_title)
            # EPUBs often contain a separate PART title page with no substantive prose.
            # Preserve it in the parse audit but never allocate quotation budget to it.
            if re.match(r"^part\b", canonical_title(chapter_title)) and chapter_words < 50:
                chapter_kind = "exclude"
            chapters.append(
                Chapter(
                    chapter_id=chapter_id,
                    title=chapter_title,
                    paragraph_ids=para_ids,
                    word_count=chapter_words,
                    kind=chapter_kind,
                )
            )

    return Book(
        title=title,
        source_path=str(source_path),
        chapters=chapters,
        paragraphs=paragraphs,
        total_words=sum(c.word_count for c in chapters if c.kind != "exclude"),
        diagnostics=diagnostics or {},
    )


def load_txt_or_md(path: Path) -> Book:
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    sections: list[tuple[str, list[tuple[str, int | None]]]] = []
    current_title = "Opening"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((current_title, [(p, None) for p in paragraph_split(text)]))
        current_lines = []

    for line in lines:
        stripped = line.strip().lstrip("#").strip()
        is_markdown_heading = bool(re.match(r"^\s{0,3}#{1,6}\s+\S", line))
        is_heading = is_markdown_heading or (len(stripped) < 120 and HEADING_RE.match(stripped))
        if is_heading:
            flush()
            current_title = stripped
        else:
            current_lines.append(line)
    flush()

    if not sections:
        sections = [("Book text", [(p, None) for p in paragraph_split(raw)])]
    return build_book_from_sections(path, path.stem, sections)


def load_docx(path: Path) -> Book:
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError("Install python-docx to read DOCX files: pip install python-docx") from exc

    doc = Document(str(path))
    sections: list[tuple[str, list[tuple[str, int | None]]]] = []
    current_title = "Opening"
    current: list[tuple[str, int | None]] = []

    def flush() -> None:
        nonlocal current
        if current:
            sections.append((current_title, current))
            current = []

    for para in doc.paragraphs:
        text = normalize_text(para.text)
        if not text:
            continue
        style_name = getattr(para.style, "name", "") or ""
        is_heading = style_name.lower().startswith("heading") or (
            len(text) < 120 and HEADING_RE.match(text)
        )
        if is_heading:
            flush()
            current_title = text
        else:
            current.append((text, None))
    flush()
    return build_book_from_sections(path, path.stem, sections)


class _EPUBBlockParser(HTMLParser):
    """Forgiving EPUB content reader with wrapper-anchor propagation."""
    TEXT_TAGS = {"p", "blockquote", "li", "td"}
    HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    WRAPPER_TAGS = {"section", "article", "div", "aside", "main", "span", "a"}
    SKIP_TAGS = {"script", "style", "nav", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.active_tag: str | None = None
        self.active_ids: set[str] = set()
        self.pending_ids: set[str] = set()
        self.buffer: list[str] = []
        self.blocks: list[tuple[str, str, set[str]]] = []
        self.semantic_tokens: set[str] = set()
        self.image_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr = {k.lower(): (v or "") for k, v in attrs}
        identifiers = {attr[key] for key in ("id", "name") if attr.get(key)}
        for key, value in attr.items():
            if key == "role" or key.endswith("}type") or key in {"epub:type", "type"}:
                self.semantic_tokens.update(value.lower().replace(":", "-").split())
        if tag == "img":
            self.image_count += 1
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in self.WRAPPER_TAGS and identifiers:
            self.pending_ids.update(identifiers)
        if tag in self.TEXT_TAGS | self.HEADING_TAGS:
            self._flush()
            self.active_tag = tag
            self.active_ids = identifiers | self.pending_ids
            self.pending_ids.clear()
            self.buffer = []
        elif tag == "br" and self.active_tag:
            self.buffer.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if not self.skip_depth and self.active_tag == tag:
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self.skip_depth and self.active_tag:
            self.buffer.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        if self.active_tag:
            value = normalize_text(unescape(" ".join(self.buffer)))
            if value:
                self.blocks.append((self.active_tag, value, set(self.active_ids)))
        self.active_tag = None
        self.active_ids = set()
        self.buffer = []


class _LooseEPUBTextParser(HTMLParser):
    """Fallback preserving order when prose is carried directly by layout containers."""
    BREAK_TAGS = {"p", "blockquote", "li", "td", "div", "section", "article"}
    HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    WRAPPER_TAGS = {"div", "section", "article", "span", "a"}
    SKIP_TAGS = {"script", "style", "nav", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.heading_tag: str | None = None
        self.heading_buffer: list[str] = []
        self.prose_buffer: list[str] = []
        self.pending_ids: set[str] = set()
        self.events: list[tuple[str, str, set[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr = {k.lower(): (v or "") for k, v in attrs}
        ids = {attr[key] for key in ("id", "name") if attr.get(key)}
        if tag in self.SKIP_TAGS:
            self._flush_prose()
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in self.WRAPPER_TAGS and ids:
            self.pending_ids.update(ids)
        if tag in self.HEADING_TAGS:
            self._flush_prose()
            self.heading_tag = tag
            self.heading_buffer = []
            self.pending_ids.update(ids)
        elif tag in self.BREAK_TAGS and not self.heading_tag:
            self._flush_prose()
            self.pending_ids.update(ids)
        elif tag == "br" and not self.heading_tag:
            self.prose_buffer.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth:
            return
        if self.heading_tag == tag:
            value = normalize_text(unescape(" ".join(self.heading_buffer)))
            if value:
                self.events.append((tag, value, set(self.pending_ids)))
            self.pending_ids.clear()
            self.heading_tag = None
            self.heading_buffer = []
        elif tag in self.BREAK_TAGS:
            self._flush_prose()

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self.heading_tag:
            self.heading_buffer.append(data)
        else:
            self.prose_buffer.append(data)

    def close(self) -> None:
        super().close()
        self._flush_prose()

    def _flush_prose(self) -> None:
        value = normalize_text(unescape(" ".join(self.prose_buffer)))
        if value and word_count(value) >= 2:
            self.events.append(("p", value, set(self.pending_ids)))
            self.pending_ids.clear()
        self.prose_buffer = []


def _zip_read_text(archive: zipfile.ZipFile, member: str) -> str:
    return archive.read(member).decode("utf-8", errors="replace")


def _epub_opf_path(archive: zipfile.ZipFile) -> str:
    try:
        root = ET.fromstring(_zip_read_text(archive, "META-INF/container.xml"))
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] == "rootfile":
                candidate = element.attrib.get("full-path")
                if candidate:
                    return candidate
    except (KeyError, ET.ParseError):
        pass
    opfs = [name for name in archive.namelist() if name.lower().endswith(".opf")]
    if not opfs:
        raise ValueError("EPUB does not contain an OPF package document.")
    return opfs[0]


def _epub_package_data(archive: zipfile.ZipFile) -> dict[str, Any]:
    opf_path = _epub_opf_path(archive)
    opf_dir = posixpath.dirname(opf_path)
    root = ET.fromstring(_zip_read_text(archive, opf_path))
    title = ""
    fixed_layout_indicators: list[str] = []
    manifest: dict[str, dict[str, str]] = {}

    for element in root.iter():
        local = element.tag.rsplit("}", 1)[-1]
        if local == "title" and element.text and not title:
            title = normalize_text(element.text)
        elif local == "meta":
            prop = element.attrib.get("property", "").lower()
            name = element.attrib.get("name", "").lower()
            value = normalize_text(element.text or element.attrib.get("content", "")).lower()
            if ("rendition:layout" in prop or "fixed-layout" in name) and "pre-paginated" in value:
                fixed_layout_indicators.append("OPF metadata declares pre-paginated/fixed layout")
        elif local == "item":
            item_id = element.attrib.get("id")
            href = element.attrib.get("href")
            if item_id and href:
                manifest[item_id] = {
                    "href": posixpath.normpath(posixpath.join(opf_dir, href)),
                    "media_type": element.attrib.get("media-type", ""),
                    "properties": element.attrib.get("properties", ""),
                }

    spine: list[str] = []
    toc_id: str | None = None
    for element in root.iter():
        local = element.tag.rsplit("}", 1)[-1]
        if local == "spine":
            toc_id = element.attrib.get("toc")
            if "pre-paginated" in element.attrib.get("properties", "").lower():
                fixed_layout_indicators.append("Spine declares pre-paginated layout")
        elif local == "itemref":
            idref = element.attrib.get("idref")
            if idref in manifest:
                spine.append(manifest[idref]["href"])

    ncx_path: str | None = manifest.get(toc_id or "", {}).get("href")
    nav_path: str | None = None
    content_docs: list[str] = []
    image_count = 0
    for item in manifest.values():
        media_type = item["media_type"].lower()
        props = set(item["properties"].lower().split())
        if "nav" in props:
            nav_path = item["href"]
        if "html" in media_type or "xhtml" in media_type:
            content_docs.append(item["href"])
        if media_type.startswith("image/"):
            image_count += 1
        if "svg" in media_type:
            fixed_layout_indicators.append("Manifest contains SVG resources")

    return {
        "title": title or "Untitled book",
        "spine": spine,
        "ncx_path": ncx_path,
        "nav_path": nav_path,
        "content_docs": content_docs,
        "image_count": image_count,
        "fixed_layout_indicators": sorted(set(fixed_layout_indicators)),
    }


def _join_href(base: str, src: str) -> tuple[str, str | None]:
    path_part, _, anchor = src.partition("#")
    return posixpath.normpath(posixpath.join(base, path_part)), (anchor or None)


def _append_nav_entry(
    mapping: dict[str, list[tuple[str | None, str]]],
    base: str,
    src: str,
    label: str,
    excluded_scope: str | None = None,
) -> None:
    if not src or not label:
        return
    href, anchor = _join_href(base, src)
    emitted = f"{excluded_scope}: {label}" if excluded_scope and classify_chapter(label) != "exclude" else label
    mapping.setdefault(href, []).append((anchor, emitted))


def _epub_ncx_map(archive: zipfile.ZipFile, toc_path: str | None) -> dict[str, list[tuple[str | None, str]]]:
    mapping: dict[str, list[tuple[str | None, str]]] = {}
    if not toc_path:
        return mapping
    try:
        root = ET.fromstring(_zip_read_text(archive, toc_path))
    except (KeyError, ET.ParseError):
        return mapping
    base = posixpath.dirname(toc_path)

    def direct_child(element: ET.Element, name: str) -> ET.Element | None:
        return next((x for x in list(element) if x.tag.rsplit("}", 1)[-1] == name), None)

    def walk(navpoint: ET.Element, excluded_scope: str | None = None) -> None:
        nav_label = direct_child(navpoint, "navLabel")
        label = ""
        if nav_label is not None:
            text_node = next((x for x in nav_label.iter() if x.tag.rsplit("}", 1)[-1] == "text"), None)
            label = normalize_text(text_node.text or "") if text_node is not None else ""
        content = direct_child(navpoint, "content")
        src = content.attrib.get("src", "") if content is not None else ""
        _append_nav_entry(mapping, base, src, label, excluded_scope)
        next_scope = label if label and classify_chapter(label) == "exclude" else excluded_scope
        for child in list(navpoint):
            if child.tag.rsplit("}", 1)[-1] == "navPoint":
                walk(child, next_scope)

    nav_map = next((x for x in root.iter() if x.tag.rsplit("}", 1)[-1] == "navMap"), None)
    if nav_map is not None:
        for child in list(nav_map):
            if child.tag.rsplit("}", 1)[-1] == "navPoint":
                walk(child)
    return mapping


def _epub_type(element: ET.Element) -> set[str]:
    values: list[str] = []
    for key, value in element.attrib.items():
        local = key.rsplit("}", 1)[-1].lower()
        if local in {"type", "role"}:
            values.extend((value or "").lower().replace(":", "-").split())
    return set(values)


def _epub3_nav_data(
    archive: zipfile.ZipFile, nav_path: str | None
) -> tuple[dict[str, list[tuple[str | None, str]]], dict[str, str]]:
    toc_map: dict[str, list[tuple[str | None, str]]] = {}
    landmarks: dict[str, str] = {}
    if not nav_path:
        return toc_map, landmarks
    try:
        root = ET.fromstring(_zip_read_text(archive, nav_path))
    except (KeyError, ET.ParseError):
        return toc_map, landmarks
    base = posixpath.dirname(nav_path)
    navs = [x for x in root.iter() if x.tag.rsplit("}", 1)[-1] == "nav"]
    toc_nav = next((n for n in navs if "toc" in _epub_type(n)), None)
    landmarks_nav = next((n for n in navs if "landmarks" in _epub_type(n)), None)

    def walk_li(element: ET.Element, mapping: dict[str, list[tuple[str | None, str]]], excluded_scope: str | None = None) -> None:
        for li in [x for x in list(element) if x.tag.rsplit("}", 1)[-1] == "li"]:
            anchor = next((x for x in list(li) if x.tag.rsplit("}", 1)[-1] == "a"), None)
            label = normalize_text(" ".join(anchor.itertext())) if anchor is not None else ""
            src = anchor.attrib.get("href", "") if anchor is not None else ""
            _append_nav_entry(mapping, base, src, label, excluded_scope)
            next_scope = label if label and classify_chapter(label) == "exclude" else excluded_scope
            for child in list(li):
                if child.tag.rsplit("}", 1)[-1] == "ol":
                    walk_li(child, mapping, next_scope)

    if toc_nav is not None:
        for child in list(toc_nav):
            if child.tag.rsplit("}", 1)[-1] == "ol":
                walk_li(child, toc_map)

    if landmarks_nav is not None:
        for anchor in [x for x in landmarks_nav.iter() if x.tag.rsplit("}", 1)[-1] == "a"]:
            src = anchor.attrib.get("href", "")
            label = normalize_text(" ".join(anchor.itertext()))
            role_tokens = _epub_type(anchor)
            href, _ = _join_href(base, src)
            semantic_label = label
            if role_tokens & {"backmatter", "bibliography", "endnotes", "rearnotes", "index", "acknowledgments", "copyright-page"}:
                semantic_label = label if classify_chapter(label) == "exclude" else f"Notes: {label}"
            if href and semantic_label:
                landmarks[href] = semantic_label
    return toc_map, landmarks


def _epub_semantic_exclusion(tokens: set[str]) -> str | None:
    mapping = {
        "backmatter": "Notes",
        "endnotes": "Notes",
        "rearnotes": "Notes",
        "footnotes": "Notes",
        "bibliography": "Bibliography",
        "index": "Index",
        "acknowledgments": "Acknowledgments",
        "copyright-page": "Copyright",
        "colophon": "Copyright",
    }
    for token, label in mapping.items():
        if token in tokens:
            return label
    return None


def _epub_html_blocks(raw: str) -> tuple[list[tuple[str, str, set[str]]], set[str], int, bool]:
    parser = _EPUBBlockParser()
    parser.feed(raw)
    parser.close()
    primary_words = sum(word_count(value) for tag, value, _ in parser.blocks if tag in parser.TEXT_TAGS)
    used_loose_fallback = False
    blocks = parser.blocks
    if primary_words < 20:
        fallback = _LooseEPUBTextParser()
        fallback.feed(raw)
        fallback.close()
        recovered_words = sum(
            word_count(value) for tag, value, _ in fallback.events if tag == "p"
        )
        if recovered_words > primary_words:
            blocks = fallback.events
            used_loose_fallback = True
    return blocks, parser.semantic_tokens, parser.image_count, used_loose_fallback


def _visible_section_title(blocks: list[tuple[str, str, set[str]]]) -> str | None:
    for tag, value, _ in blocks[:10]:
        if tag in {"h1", "h2", "h3"} and word_count(value) <= 24:
            return value
        if tag in {"p", "li"} and word_count(value) <= 16 and HEADING_RE.match(value):
            return value
    return None


def _split_blocks_by_navigation(
    blocks: list[tuple[str, str, set[str]]],
    toc_entries: list[tuple[str | None, str]],
) -> list[tuple[str, list[tuple[str, int | None]]]]:
    readable_tags = _EPUBBlockParser.TEXT_TAGS
    anchor_to_title = {anchor: label for anchor, label in toc_entries if anchor}
    if len(toc_entries) <= 1 or not anchor_to_title:
        readable = [(value, None) for tag, value, _ in blocks if tag in readable_tags and word_count(value) >= 2]
        return [(toc_entries[0][1], readable)] if readable else []
    sections: list[tuple[str, list[tuple[str, int | None]]]] = []
    current_title = toc_entries[0][1]
    current_blocks: list[tuple[str, int | None]] = []
    for tag, value, ids in blocks:
        matching = next((anchor_to_title[a] for a in ids if a in anchor_to_title), None)
        if matching and current_blocks:
            sections.append((current_title, current_blocks))
            current_blocks = []
            current_title = matching
        elif matching:
            current_title = matching
        if tag in readable_tags and word_count(value) >= 2:
            current_blocks.append((value, None))
    if current_blocks:
        sections.append((current_title, current_blocks))
    return sections


def load_epub(path: Path) -> Book:
    sections: list[tuple[str, list[tuple[str, int | None]]]] = []
    pending_front: list[tuple[str, int | None]] = []
    active_excluded_scope: str | None = None
    fallback_items = 0
    loose_prose_items = 0
    unassigned_words = 0
    total_images_in_content = 0
    warnings: list[str] = []

    with zipfile.ZipFile(path) as archive:
        package = _epub_package_data(archive)
        title = package["title"]
        spine = package["spine"] or package["content_docs"]
        epub3_toc, landmarks = _epub3_nav_data(archive, package["nav_path"])
        ncx_toc = _epub_ncx_map(archive, package["ncx_path"])
        toc_map = epub3_toc or ncx_toc
        navigation_source = "EPUB3 nav.xhtml" if epub3_toc else ("NCX" if ncx_toc else "visible-heading fallback")

        for member in spine:
            if member not in archive.namelist():
                warnings.append(f"Spine resource missing from archive: {member}")
                continue
            blocks, semantic_tokens, image_count, used_loose = _epub_html_blocks(_zip_read_text(archive, member))
            total_images_in_content += image_count
            if used_loose:
                loose_prose_items += 1
            readable = [(value, None) for tag, value, _ in blocks if tag in _EPUBBlockParser.TEXT_TAGS and word_count(value) >= 2]
            if not readable:
                continue

            semantic_label = _epub_semantic_exclusion(semantic_tokens) or landmarks.get(member)
            toc_entries = toc_map.get(member, [])
            if semantic_label and classify_chapter(semantic_label) == "exclude":
                sections.append((semantic_label, readable))
                active_excluded_scope = semantic_label
                continue

            if toc_entries:
                recovered = _split_blocks_by_navigation(blocks, toc_entries)
                sections.extend(recovered)
                active_excluded_scope = (
                    toc_entries[-1][1] if recovered and classify_chapter(toc_entries[-1][1]) == "exclude" else None
                )
                continue

            if active_excluded_scope and sections:
                sections[-1][1].extend(readable)
                continue

            fallback_items += 1
            recovered_sections: list[tuple[str, list[tuple[str, int | None]]]] = []
            current_title: str | None = None
            current_blocks: list[tuple[str, int | None]] = []
            for tag, value, _ in blocks:
                heading_like = (
                    (tag in {"h1", "h2", "h3"} and word_count(value) <= 24)
                    or (tag in {"p", "li"} and word_count(value) <= 16 and HEADING_RE.match(value))
                )
                if heading_like:
                    if current_title and current_blocks:
                        recovered_sections.append((current_title, current_blocks))
                    current_title = value
                    current_blocks = []
                elif tag in _EPUBBlockParser.TEXT_TAGS and word_count(value) >= 2:
                    current_blocks.append((value, None))
            if current_title and current_blocks:
                recovered_sections.append((current_title, current_blocks))

            if recovered_sections:
                sections.extend(recovered_sections)
            elif sections:
                sections[-1][1].extend(readable)
            else:
                pending_front.extend(readable)
                unassigned_words += sum(word_count(value) for value, _ in readable)

        if pending_front:
            sections.insert(0, ("Unmapped front matter", pending_front))

    if not sections:
        raise ValueError("EPUB parsing returned no readable content.")

    extracted_words = sum(word_count(value) for _, blocks in sections for value, _ in blocks)
    fixed_indicators = package["fixed_layout_indicators"]
    image_only_risk = "low"
    if extracted_words < 500 and (package["image_count"] > 5 or total_images_in_content > 5):
        image_only_risk = "high"
    elif fixed_indicators or (total_images_in_content > len(spine) and extracted_words < 5000):
        image_only_risk = "medium"
    if loose_prose_items:
        warnings.append(f"Recovered prose from nonstandard container markup in {loose_prose_items} spine item(s).")
    if fallback_items:
        warnings.append(f"Used visible-heading/continuation recovery in {fallback_items} spine item(s).")
    if fixed_indicators:
        warnings.append("Fixed-layout/SVG indicators detected; verify the structure report and sample text.")
    proceed = image_only_risk != "high" and extracted_words >= 500
    confidence = "high"
    if navigation_source == "visible-heading fallback" or loose_prose_items or image_only_risk == "medium":
        confidence = "medium"
    if not proceed:
        confidence = "low"

    diagnostics = {
        "format": "EPUB",
        "navigation_source": navigation_source,
        "spine_items_found": len(spine),
        "sections_recovered_before_classification": len(sections),
        "unassigned_text_words": unassigned_words,
        "prose_fallback_items": loose_prose_items,
        "fixed_layout_indicators": fixed_indicators,
        "manifest_image_resources": package["image_count"],
        "content_image_elements": total_images_in_content,
        "image_only_content_risk": image_only_risk,
        "parser_confidence": confidence,
        "proceed_with_llm_extraction": proceed,
        "warnings": warnings,
    }
    return build_book_from_sections(path, title, sections, diagnostics=diagnostics)


PDF_BOILERPLATE_RE = re.compile(
    r"(Downloaded from\s+https?://academic\.oup\.com/\S+.*?$|"
    r"The Edge of Sentience:.*?Oxford University Press\..*?$|"
    r"©\s*Jonathan Birch.*?$|"
    r"DOI:\s*\S+.*?$)",
    flags=re.IGNORECASE | re.MULTILINE,
)
PDF_MAIN_TITLE_RE = re.compile(
    r"^(summary of the framework.*|introduction|preface|prologue|foreword|"
    r"\d+\s+\S.*|conclusion|epilogue|afterword|bibliography|references|index|"\
    r"list of interviews.*|written correspondence.*|photo[- ]insert.*|illustrations|plates)$",
    flags=re.IGNORECASE,
)


def normalize_pdf_line(line: str) -> str:
    line = line.replace("\u00ad", "").replace("\ufb01", "fi").replace("\ufb02", "fl")
    line = line.replace("\ufffd", "")
    line = re.sub(r"\s+", " ", line).strip()
    return line


def normalize_repeated_line(line: str) -> str:
    line = normalize_pdf_line(line).lower()
    line = re.sub(r"\d+", "#", line)
    return re.sub(r"[^a-z# ]+", "", line).strip()


def pdf_margin_noise_lines(doc: Any) -> set[str]:
    """Identify short running headers/footers recurring across PDF pages."""
    counts: Counter[str] = Counter()
    for page in doc:
        lines = [normalize_pdf_line(x) for x in page.get_text("text", sort=True).splitlines()]
        marginal = lines[:3] + lines[-4:]
        for line in marginal:
            key = normalize_repeated_line(line)
            if key and len(key) <= 100:
                counts[key] += 1
    threshold = max(3, int(len(doc) * 0.06))
    return {line for line, count in counts.items() if count >= threshold}


def clean_pdf_block(raw: str, recurring_noise: set[str]) -> str:
    raw = raw.replace("\u00ad", "").replace("\ufb01", "fi").replace("\ufb02", "fl")
    raw = re.sub(r"(?<=\w)[\ufffd\ufffe](?=\w)", "-", raw)
    raw = raw.replace("\ufffd", "").replace("\ufffe", "")
    raw = PDF_BOILERPLATE_RE.sub("", raw)
    lines: list[str] = []
    for candidate in raw.splitlines():
        line = normalize_pdf_line(candidate)
        if not line:
            continue
        if re.match(r"^\d+$", line):
            continue
        if normalize_repeated_line(line) in recurring_noise:
            continue
        if line.lower().startswith("downloaded from https://academic.oup.com"):
            continue
        lines.append(line)
    if not lines:
        return ""

    # Restore prose split by PDF line wraps while repairing line-break hyphenation.
    text = lines[0]
    for line in lines[1:]:
        if text.endswith("-") and line and line[0].islower():
            text = text[:-1] + line
        else:
            text += " " + line
    text = re.sub(r"(?<=\w)\s*[-‐‑]\s+(?=\w)", "-", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return normalize_text(text)


def load_manual_chapter_map(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--chapter-map must be a JSON list of objects.")
    for entry in data:
        if not all(k in entry for k in ("title", "start_page")):
            raise ValueError("Every chapter-map entry must contain title and start_page.")
    return data


def pdf_outline_sections(doc: Any) -> list[dict[str, Any]]:
    toc = doc.get_toc(simple=True)
    if not toc:
        return []
    candidates: list[dict[str, Any]] = []
    for level, title, page in toc:
        title = normalize_text(title)
        if PDF_MAIN_TITLE_RE.match(title):
            candidates.append({"level": level, "title": title, "start_page": int(page)})
    # Prefer the outline level with a plausible sequence of book chapters.
    by_level: dict[int, list[dict[str, Any]]] = {}
    for item in candidates:
        by_level.setdefault(item["level"], []).append(item)
    plausible = [items for items in by_level.values() if len(items) >= 3]
    chosen = max(plausible, key=len) if plausible else candidates
    chosen = sorted({(x["title"], x["start_page"]): x for x in chosen}.values(), key=lambda x: x["start_page"])
    return chosen



def pdf_detect_sections_from_text(doc: Any, recurring_noise: set[str]) -> list[dict[str, Any]]:
    """Recover chapter starts from visible page text when a PDF outline is absent."""
    found: list[dict[str, Any]] = []
    for page_no, page in enumerate(doc, start=1):
        lines = [normalize_pdf_line(line) for line in page.get_text("text", sort=True).splitlines()]
        candidates = [
            line for line in lines[:12]
            if line and normalize_repeated_line(line) not in recurring_noise and len(line) <= 140
        ]
        for candidate in candidates:
            if PDF_MAIN_TITLE_RE.match(candidate) or HEADING_RE.match(candidate):
                found.append({"title": candidate, "start_page": page_no})
                break
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in found:
        key = (canonical_title(str(item["title"])), int(item["start_page"]))
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped if len(deduped) >= 3 else []


def load_pdf(path: Path, chapter_map_path: Path | None = None) -> Book:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("Install PyMuPDF to read PDFs robustly: pip install pymupdf") from exc

    doc = fitz.open(str(path))
    metadata = doc.metadata or {}
    title = str(metadata.get("title") or path.stem)
    recurring_noise = pdf_margin_noise_lines(doc)
    outline = load_manual_chapter_map(chapter_map_path) if chapter_map_path else pdf_outline_sections(doc)
    if not outline:
        outline = pdf_detect_sections_from_text(doc, recurring_noise)

    if not outline:
        visible_words = sum(word_count(page.get_text("text", sort=True) or "") for page in doc)
        if visible_words < 500:
            raise ValueError(
                "This PDF contains little or no extractable text. Run OCR first, then re-run the condenser."
            )
        raise ValueError(
            "No reliable chapter outline could be recovered from PDF bookmarks or visible headings. "
            "Supply --chapter-map chapters.json. The condenser will not spend API calls on an "
            "undifferentiated full-book extraction."
        )

    starts = [int(x["start_page"]) for x in outline]
    sections: list[tuple[str, list[tuple[str, int | None]]]] = []
    for idx, entry in enumerate(outline):
        start_page = int(entry["start_page"])
        end_page = int(entry.get("end_page") or ((starts[idx + 1] - 1) if idx + 1 < len(starts) else len(doc)))
        title_text = str(entry["title"]).strip()
        blocks: list[tuple[str, int | None]] = []
        for page_no in range(start_page, min(end_page, len(doc)) + 1):
            page = doc[page_no - 1]
            for block in page.get_text("blocks", sort=True):
                cleaned = clean_pdf_block(str(block[4]), recurring_noise)
                if cleaned and word_count(cleaned) >= 3:
                    blocks.append((cleaned, page_no))
        if blocks:
            sections.append((title_text, blocks))

    if not sections:
        raise ValueError("PDF parsing returned no readable chapter content.")
    extracted_words = sum(word_count(value) for _, blocks in sections for value, _ in blocks)
    page_image_count = sum(len(page.get_images(full=True)) for page in doc)
    source = "manual chapter map" if chapter_map_path else ("PDF bookmarks" if pdf_outline_sections(doc) else "visible-heading fallback")
    image_risk = "high" if extracted_words < 500 and page_image_count > len(doc) * 0.5 else ("medium" if page_image_count > len(doc) and extracted_words < 5000 else "low")
    diagnostics = {
        "format": "PDF",
        "navigation_source": source,
        "pages": len(doc),
        "sections_recovered_before_classification": len(sections),
        "unassigned_text_words": 0,
        "fixed_layout_indicators": ["PDF pages are fixed-layout by definition"],
        "content_image_elements": page_image_count,
        "image_only_content_risk": image_risk,
        "parser_confidence": "low" if image_risk == "high" else ("medium" if source == "visible-heading fallback" or image_risk == "medium" else "high"),
        "proceed_with_llm_extraction": image_risk != "high" and extracted_words >= 500,
        "warnings": ["Verify visible-heading recovery before full extraction."] if source == "visible-heading fallback" else [],
    }
    return build_book_from_sections(path, title, sections, diagnostics=diagnostics)


def load_book(path: Path, chapter_map_path: Path | None = None) -> Book:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return load_txt_or_md(path)
    if suffix == ".docx":
        return load_docx(path)
    if suffix == ".epub":
        return load_epub(path)
    if suffix == ".pdf":
        return load_pdf(path, chapter_map_path=chapter_map_path)
    raise ValueError(f"Unsupported input type: {suffix}. Use EPUB, PDF, DOCX, TXT, or MD.")


def validate_book_structure(book: Book) -> None:
    included = [c for c in book.chapters if c.kind != "exclude" and c.word_count > 0]
    diagnostics = book.diagnostics or {}
    if diagnostics.get("proceed_with_llm_extraction") is False:
        raise ValueError(
            "Parser confidence is insufficient for extractive condensation: the source may be fixed-layout, "
            "image-based, or insufficiently recoverable as prose. Review parsed_structure_report.md; "
            "use OCR, a better EPUB export, or a manual PDF chapter map before proceeding."
        )
    excluded_words = sum(c.word_count for c in book.chapters if c.kind == "exclude")
    all_words = sum(c.word_count for c in book.chapters)
    if book.total_words <= 0:
        raise ValueError(
            "No included book text remains after section classification. The source structure was likely "
            "misread or every section was mistaken for back matter. Review parsed_structure_report.md."
        )
    if book.total_words > 30000 and len(included) < 3:
        raise ValueError(
            f"Structural parsing is implausible: {book.total_words:,} words were assigned to only "
            f"{len(included)} included section(s). Provide a chapter map for PDF input or inspect the EPUB navigation."
        )
    if all_words > 0 and excluded_words / all_words > 0.50:
        raise ValueError(
            f"Structural parsing is implausible: {excluded_words / all_words:.1%} of extracted words were "
            "classified as back matter. Review parsed_structure_report.md before proceeding."
        )
    if excluded_words > 0:
        LOGGER.info("Excluded non-reading matter was detected and will not be used for passage selection.")


# ---------------------------------------------------------------------------
# LLM interface
# ---------------------------------------------------------------------------

class LLM:
    def __init__(self, model: str, retries: int = 3) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it in your environment before running the full pipeline, "
                "or use --parse-only to validate a source without API calls."
            )
        self.client = OpenAI()
        self.model = model
        self.retries = retries

    def structured(self, instructions: str, user_input: str, response_model: type[T]) -> T:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": user_input},
                    ],
                    text_format=response_model,
                )
                parsed = response.output_parsed
                if parsed is None:
                    raise RuntimeError("The API returned no parsed structured output.")
                return parsed
            except Exception as exc:  # API/network/schema error handling with limited retries
                last_error = exc
                if attempt == self.retries:
                    break
                delay = min(2 ** attempt, 10)
                LOGGER.warning("LLM call failed on attempt %d/%d: %s. Retrying in %ds.", attempt, self.retries, exc, delay)
                time.sleep(delay)
        raise RuntimeError(f"LLM structured-output call failed after {self.retries} attempts: {last_error}") from last_error


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def render_paragraphs(book: Book, paragraph_ids: Sequence[str], max_words: int | None = None) -> str:
    rendered = []
    used = 0
    for pid in paragraph_ids:
        para = book.paragraphs[pid]
        if max_words is not None and used + para.word_count > max_words:
            break
        page = f" [p. {para.page}]" if para.page is not None else ""
        rendered.append(f"{para.paragraph_id}{page}: {para.text}")
        used += para.word_count
    return "\n\n".join(rendered)


def overview_input(book: Book, emphasis: str, max_structural_words: int) -> str:
    chapter_manifest = "\n".join(
        f"- {c.chapter_id}: {c.title} ({c.word_count} words; kind={c.kind})"
        for c in book.chapters
        if c.kind != "exclude"
    )
    front = [c for c in book.chapters if c.kind == "front"]
    ending = [c for c in book.chapters if c.kind == "ending"]

    if not front:
        front = [c for c in book.chapters if c.kind == "body"][:1]
    if not ending:
        ending = [c for c in book.chapters if c.kind == "body"][-1:]

    per_section_limit = max(1000, max_structural_words // max(1, len(front) + len(ending)))
    structural_texts: list[str] = []
    for chapter in front + ending:
        excerpt = render_paragraphs(book, chapter.paragraph_ids, max_words=per_section_limit)
        structural_texts.append(f"\n## {chapter.chapter_id}: {chapter.title}\n{excerpt}")

    return f"""BOOK TITLE: {book.title}
TOTAL INCLUDED SOURCE WORDS: {book.total_words}
DESIRED READING EMPHASIS: {emphasis}

CHAPTER MANIFEST:
{chapter_manifest}

INTRODUCTION / OPENING AND ENDING MATERIAL:
{''.join(structural_texts)}
"""


def split_paragraph_ids_by_words(book: Book, paragraph_ids: Sequence[str], max_words: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    for pid in paragraph_ids:
        wc = book.paragraphs[pid].word_count
        if current and current_words + wc > max_words:
            chunks.append(current)
            current = []
            current_words = 0
        current.append(pid)
        current_words += wc
    if current:
        chunks.append(current)
    return chunks


OVERVIEW_SYSTEM = """You are the structural editor for an extractive abridgement of a rights-cleared nonfiction book.
Your output is analytical metadata only; exact original prose will be selected later.

First classify the work into exactly one nonfiction form:
argumentative, historical_investigative_narrative, biography_memoir, technical_explanatory,
case_based_policy_business, or mixed.

Study the section list, opening material, and ending material. Identify the governing question and thesis,
then map the chronological or logical arc in reading order. Determine what each chapter contributes and
assign selection priorities. For historical or investigative narrative, prioritize chronological development,
institutional change, principal actors only where they drive events, turning points, causes, consequences,
and the author's final interpretation. For argumentative or technical works, prioritize claims, concepts,
mechanisms, evidence, objections, and conclusions. Use chapter IDs exactly as supplied. Do not allow early
setup chapters or the concluding interpretation to disappear merely because middle episodes are dramatic."""

BASE_CHAPTER_SYSTEM = """You are selecting source passages for a full extractive abridgement of a rights-cleared nonfiction book.
Return paragraph identifiers only; never quote, paraphrase, rewrite, merge, or complete source text.
The application retrieves exact original wording after your selection.

READABILITY IS A HARD CONSTRAINT. Each selected block must be a self-contained reading passage, normally
200 to 1,200 words. Start where the author introduces a claim, event, case, mechanism, or interpretation;
do not start mid-response, mid-anecdote, or with unresolved referents. End after the local point or event is
completed. Prefer several compact complete passages over one long episode. Do not select bibliography,
reference material, interviews lists, photo inserts, captions, publisher notices, or page furniture.

Each selected block must be assigned one function: setup, definition, claim, mechanism, evidence,
representative_episode, turning_point, consequence, counterargument, interpretation, or conclusion.

This is the candidate stage. Nominate approximately the requested share of the supplied chapter/chunk.
A later global stage controls balance, removes redundancy, and enforces the final book-level word budget.
Every block ID must begin with the supplied prefix. Every paragraph ID must come from the supplied input."""

SCORING_SYSTEM = """You are ranking candidate quotation blocks for a readable extractive abridgement.
Your output is analytical metadata only; do not reproduce quotation text. Score each supplied block exactly
once. Reward passages that are necessary to reconstruct the book's chronological or logical arc, establish
causes or arguments, mark decisive turns or conclusions, and read coherently when extracted. Penalize blocks
that duplicate another passage, contain excessive episode-level detail relative to their importance, or require
omitted context to make sense. Identify substantial redundancy using candidate block IDs."""

QC_SYSTEM = """You are performing final editorial quality control on a proposed extractive abridgement.
Do not rewrite quotations and do not invent block IDs. Check whether the retained passages represent the
book's complete chronological or logical arc; preserve setup, decisive developments, consequences, and
conclusion. Recommend removals only for genuine redundancy, excessive detail, or unreadability, and additions
only for missing coverage. A protected chapter anchor should not be removed unless another retained block from
that same chapter already preserves its essential function. Transition notes are audit comments only and must
not appear in the reading edition."""


def mode_guidance(nonfiction_form: str) -> str:
    if nonfiction_form == "historical_investigative_narrative":
        return """This book is historical/investigative narrative. Preserve chronology and causal development.
Select compact passages that establish institutional origins, major program shifts, decisive episodes,
consequences, and retrospective interpretation. A vivid episode is valuable only insofar as it advances the
historical arc. Penalize extended scene-setting and repeated detail once the significance is established."""
    if nonfiction_form == "biography_memoir":
        return """This book is biographical or memoir-based. Preserve life chronology, formative episodes,
relationships that alter decisions, turning points, and retrospective interpretation. Do not over-retain
atmospheric scenes that do not change the subject's trajectory."""
    if nonfiction_form == "technical_explanatory":
        return """This book is technical/explanatory. Preserve definitions, mechanisms, procedures, key evidence,
representative examples needed for understanding, limitations, and conclusions. Deprioritize repeated
illustrations once a mechanism is clear."""
    if nonfiction_form == "case_based_policy_business":
        return """This book is case-based policy/business nonfiction. Preserve the framework, decisions,
representative cases, outcomes, limitations, and implications. Avoid accumulating multiple cases that make
the same point."""
    if nonfiction_form == "argumentative":
        return """This book is argumentative nonfiction. Preserve thesis, core claims, definitions, supporting
evidence, substantive objections, responses, and conclusion. Deprioritize rhetorical illustration."""
    return """This book has a mixed nonfiction form. Preserve its governing arc while balancing argument,
chronology, evidence, mechanisms, and conclusions. Avoid allowing vivid local material to displace the whole."""


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def create_structural_overview(
    llm: LLM,
    book: Book,
    emphasis: str,
    max_structural_words: int,
    output_dir: Path,
) -> StructuralOverview:
    LOGGER.info("Generating structural overview.")
    user_input = overview_input(book, emphasis, max_structural_words)
    overview = llm.structured(OVERVIEW_SYSTEM, user_input, StructuralOverview)
    safe_json_dump(overview, output_dir / "structural_overview.json")
    return overview


def candidate_text_budget(chapter: Chapter, candidate_ratio: float) -> int:
    return max(120, int(chapter.word_count * candidate_ratio))


def select_candidates_for_chapter(
    llm: LLM,
    book: Book,
    chapter: Chapter,
    overview: StructuralOverview,
    emphasis: str,
    candidate_ratio: float,
    chapter_chunk_words: int,
    output_dir: Path,
) -> list[SelectedBlock]:
    chunks = split_paragraph_ids_by_words(book, chapter.paragraph_ids, chapter_chunk_words)
    blocks: list[SelectedBlock] = []
    chapter_function_notes: list[str] = []
    omitted_notes: list[str] = []

    LOGGER.info("Selecting candidate blocks for %s (%d chunk(s)).", chapter.title, len(chunks))
    for chunk_index, para_ids in enumerate(chunks, start=1):
        chunk_words = sum(book.paragraphs[pid].word_count for pid in para_ids)
        requested_words = max(100, int(chunk_words * candidate_ratio))
        prefix = f"{chapter.chapter_id}-K{chunk_index:02d}-B"
        input_text = f"""BOOK STRUCTURAL OVERVIEW:
{overview.model_dump_json(indent=2)}

READING EMPHASIS:
{emphasis}

NONFICTION-FORM GUIDANCE:
{mode_guidance(overview.nonfiction_form)}

CHAPTER:
{chapter.chapter_id}: {chapter.title}

CHUNK:
{chunk_index} of {len(chunks)}; {chunk_words} words

CANDIDATE EXTRACTION TARGET:
Nominate approximately {requested_words} words of paragraph blocks from this chunk, corresponding to about {candidate_ratio:.0%} of the source chunk.

REQUIRED BLOCK-ID PREFIX:
{prefix}

SOURCE PARAGRAPHS:
{render_paragraphs(book, para_ids)}
"""
        response = llm.structured(BASE_CHAPTER_SYSTEM + "\n\n" + mode_guidance(overview.nonfiction_form), input_text, ChapterCandidateResponse)
        if response.chapter_id != chapter.chapter_id:
            LOGGER.warning("Response chapter ID %s differed from requested %s; using requested ID.", response.chapter_id, chapter.chapter_id)
        chapter_function_notes.append(response.chapter_contribution)
        omitted_notes.extend(response.omitted_material_description)

        para_set = set(para_ids)
        source_order = {pid: ix for ix, pid in enumerate(para_ids)}
        output_block_counter = 0
        for block in response.candidate_blocks:
            valid_ids = [pid for pid in block.paragraph_ids if pid in para_set]
            if not valid_ids:
                LOGGER.warning("Ignoring block with no valid paragraph IDs: %s", block.block_id)
                continue

            ordered_ids = sorted(set(valid_ids), key=lambda pid: source_order[pid])

            # A quotation block must be contiguous in the original source. If the
            # model nominates separated paragraphs, preserve integrity by splitting
            # them into independent blocks rather than silently removing intervening text.
            runs: list[list[str]] = []
            current_run: list[str] = []
            previous_position: int | None = None
            for pid in ordered_ids:
                position = source_order[pid]
                if current_run and previous_position is not None and position != previous_position + 1:
                    runs.append(current_run)
                    current_run = []
                current_run.append(pid)
                previous_position = position
            if current_run:
                runs.append(current_run)

            for run_index, contiguous_ids in enumerate(runs, start=1):
                output_block_counter += 1
                exact_text = "\n\n".join(book.paragraphs[pid].text for pid in contiguous_ids)
                reason = block.selection_reason
                if len(runs) > 1:
                    reason = f"{reason} [Split into contiguous source block {run_index} of {len(runs)}.]"
                blocks.append(
                    SelectedBlock(
                        block_id=f"{prefix}{output_block_counter:02d}",
                        chapter_id=chapter.chapter_id,
                        chapter_title=chapter.title,
                        paragraph_ids=contiguous_ids,
                        word_count=sum(book.paragraphs[pid].word_count for pid in contiguous_ids),
                        selection_reason=reason,
                        importance=block.importance,
                        themes=block.themes,
                        redundancy_risk=block.redundancy_risk,
                        block_function=block.block_function,
                        text=exact_text,
                    )
                )

    payload = {
        "chapter_id": chapter.chapter_id,
        "chapter_title": chapter.title,
        "chapter_contribution_notes": chapter_function_notes,
        "omitted_material_description": omitted_notes,
        "candidate_blocks": [asdict(b) for b in blocks],
    }
    safe_json_dump(payload, output_dir / "chapter_candidates" / f"{chapter.chapter_id}.json")
    return blocks


def all_candidate_blocks(
    llm: LLM,
    book: Book,
    overview: StructuralOverview,
    emphasis: str,
    candidate_ratio: float,
    chapter_chunk_words: int,
    output_dir: Path,
) -> list[SelectedBlock]:
    included = [c for c in book.chapters if c.kind != "exclude"]
    candidates: list[SelectedBlock] = []
    for chapter in included:
        candidates.extend(
            select_candidates_for_chapter(
                llm=llm,
                book=book,
                chapter=chapter,
                overview=overview,
                emphasis=emphasis,
                candidate_ratio=candidate_ratio,
                chapter_chunk_words=chapter_chunk_words,
                output_dir=output_dir,
            )
        )
    return candidates


def block_preview(block: SelectedBlock, preview_words: int = 180) -> str:
    return truncate_words(block.text, preview_words)


def score_candidates(
    llm: LLM,
    candidates: list[SelectedBlock],
    overview: StructuralOverview,
    score_batch_size: int,
    output_dir: Path,
) -> list[SelectedBlock]:
    LOGGER.info("Scoring %d candidate blocks.", len(candidates))
    manifest = "\n".join(
        f"{b.block_id} | {b.chapter_id} | words={b.word_count} | themes={', '.join(b.themes)} | reason={b.selection_reason}"
        for b in candidates
    )
    scores_by_id: dict[str, ScoredBlockResponse] = {}

    for start in range(0, len(candidates), score_batch_size):
        batch = candidates[start : start + score_batch_size]
        supplied = "\n\n".join(
            f"BLOCK {b.block_id}\nChapter: {b.chapter_id}: {b.chapter_title}\nWords: {b.word_count}\n"
            f"Function: {b.block_function}\nInitial importance: {b.importance}\nReason: {b.selection_reason}\nThemes: {', '.join(b.themes)}\n"
            f"Source preview:\n{block_preview(b)}"
            for b in batch
        )
        user_input = f"""STRUCTURAL OVERVIEW:
{overview.model_dump_json(indent=2)}

COMPLETE CANDIDATE MANIFEST FOR REDUNDANCY REFERENCE:
{truncate_words(manifest, 7000)}

NONFICTION-FORM GUIDANCE:
{mode_guidance(overview.nonfiction_form)}

SCORING RUBRIC:
Block value = 3*chronological_or_logical_necessity
              + 3*institutional_causal_or_argumentative_importance
              + 2*turning_point_or_conclusion_value
              + 2*explanatory_density + readability
              - 2*redundancy_penalty - 2*excessive_detail_penalty
              - context_dependence_penalty.

BLOCKS TO SCORE IN THIS CALL:
{supplied}
"""
        response = llm.structured(SCORING_SYSTEM, user_input, ScoringResponse)
        for score in response.scores:
            scores_by_id[score.block_id] = score

    for block in candidates:
        score = scores_by_id.get(block.block_id)
        if score is None:
            LOGGER.warning("No score returned for %s; assigning neutral score.", block.block_id)
            block.score = 10.0
            continue
        block.score = (
            3 * score.chronological_or_logical_necessity
            + 3 * score.institutional_causal_or_argumentative_importance
            + 2 * score.turning_point_or_conclusion_value
            + 2 * score.explanatory_density
            + score.readability
            - 2 * score.redundancy_penalty
            - 2 * score.excessive_detail_penalty
            - score.context_dependence_penalty
        )
        block.redundant_with = score.redundant_with

    safe_json_dump([asdict(b) for b in candidates], output_dir / "scored_candidates.json")
    return candidates


def chapter_priority_map(overview: StructuralOverview) -> dict[str, str]:
    return {item.chapter_id: item.priority.lower() for item in overview.chapter_priorities}


def choose_blocks_under_budget(
    book: Book,
    candidates: list[SelectedBlock],
    overview: StructuralOverview,
    target_ratio: float,
    coverage_mode: str = "all",
    chapter_max_share: float = 0.08,
) -> tuple[list[SelectedBlock], int]:
    """Select readable blocks while protecting book-wide coverage and preventing episode dominance."""
    target_words = int(book.total_words * target_ratio)
    priority = chapter_priority_map(overview)
    chosen: list[SelectedBlock] = []
    chosen_ids: set[str] = set()
    chosen_words_by_chapter: dict[str, int] = {}
    used_words = 0
    chapter_cap = max(600, int(target_words * chapter_max_share))

    included_chapters = [c for c in book.chapters if c.kind != "exclude"]
    by_chapter: dict[str, list[SelectedBlock]] = {}
    for block in candidates:
        by_chapter.setdefault(block.chapter_id, []).append(block)

    def priority_multiplier(block: SelectedBlock) -> float:
        return {"high": 1.18, "medium": 1.0, "low": 0.86}.get(priority.get(block.chapter_id, "medium"), 1.0)

    def utility(block: SelectedBlock) -> float:
        continuity_bonus = 1.07 if block.block_function in {
            "setup", "turning_point", "consequence", "interpretation", "conclusion"
        } else 1.0
        return priority_multiplier(block) * continuity_bonus * block.score / math.sqrt(max(block.word_count, 1))

    def cap_for(block: SelectedBlock, anchor: bool = False) -> int:
        # The concluding interpretation and very short chapters may exceed the generic cap only
        # when one anchor block is needed to preserve coverage.
        if anchor and block.word_count > chapter_cap:
            return block.word_count
        if priority.get(block.chapter_id) == "high":
            return int(chapter_cap * 1.15)
        return chapter_cap

    def conflicts(block: SelectedBlock) -> bool:
        return any(r in chosen_ids for r in block.redundant_with)

    def can_add(block: SelectedBlock, *, anchor: bool = False, allow_overshoot: bool = False) -> bool:
        if block.block_id in chosen_ids or conflicts(block):
            return False
        budget_limit = int(target_words * (1.03 if allow_overshoot else 1.0))
        if used_words + block.word_count > budget_limit:
            return False
        current = chosen_words_by_chapter.get(block.chapter_id, 0)
        return current + block.word_count <= cap_for(block, anchor=anchor)

    def add(block: SelectedBlock, protected: bool = False) -> None:
        nonlocal used_words
        block.protected_anchor = protected
        chosen.append(block)
        chosen_ids.add(block.block_id)
        used_words += block.word_count
        chosen_words_by_chapter[block.chapter_id] = chosen_words_by_chapter.get(block.chapter_id, 0) + block.word_count

    # First pass: ensure the reading edition contains the full arc rather than only dramatic chapters.
    for chapter in included_chapters:
        available = by_chapter.get(chapter.chapter_id, [])
        if not available:
            continue
        must_cover = coverage_mode == "all" or (
            coverage_mode == "major" and (
                chapter.kind in {"front", "ending"} or priority.get(chapter.chapter_id, "medium") in {"high", "medium"}
            )
        )
        if not must_cover:
            continue
        candidates_for_anchor = sorted(
            available,
            key=lambda b: (utility(b), -b.word_count),
            reverse=True,
        )
        for anchor in candidates_for_anchor:
            if can_add(anchor, anchor=True, allow_overshoot=True):
                add(anchor, protected=True)
                break

    # Second pass: select additional high-value blocks, subject to per-chapter concentration limits.
    ranked = sorted(candidates, key=utility, reverse=True)
    for block in ranked:
        if can_add(block):
            add(block)

    # Third pass: fill only a substantial residual budget while respecting chapter caps.
    if used_words < int(target_words * 0.94):
        additions = [b for b in ranked if b.block_id not in chosen_ids and not conflicts(b)]
        additions.sort(key=lambda b: (abs((used_words + b.word_count) - target_words), -utility(b)))
        for block in additions:
            if can_add(block, allow_overshoot=True):
                add(block)
            if used_words >= int(target_words * 0.97):
                break

    chapter_order = {c.chapter_id: i for i, c in enumerate(book.chapters)}
    para_order = {pid: p.index for pid, p in book.paragraphs.items()}
    chosen.sort(key=lambda b: (chapter_order.get(b.chapter_id, 10**6), para_order[b.paragraph_ids[0]]))
    return chosen, target_words


def quality_control(
    llm: LLM,
    overview: StructuralOverview,
    selected: list[SelectedBlock],
    candidates: list[SelectedBlock],
    target_words: int,
    output_dir: Path,
) -> QualityControlResponse:
    selected_ids = {b.block_id for b in selected}
    retained_words = sum(b.word_count for b in selected)
    retained_manifest = "\n\n".join(
        f"RETAINED {b.block_id} | {b.chapter_id}: {b.chapter_title} | words={b.word_count} | score={b.score:.1f}\n"
        f"Reason: {b.selection_reason}\nPreview: {block_preview(b, 120)}"
        for b in selected
    )
    alternatives = sorted(
        [b for b in candidates if b.block_id not in selected_ids],
        key=lambda x: x.score,
        reverse=True,
    )[:20]
    alternative_manifest = "\n".join(
        f"AVAILABLE {b.block_id} | {b.chapter_id}: {b.chapter_title} | words={b.word_count} | score={b.score:.1f} | {b.selection_reason}"
        for b in alternatives
    )

    input_text = f"""STRUCTURAL OVERVIEW:
{overview.model_dump_json(indent=2)}

WORD BUDGET:
Target: {target_words}
Currently retained: {retained_words}

PROPOSED RETAINED BLOCKS:
{truncate_words(retained_manifest, 12000)}

HIGH-SCORING UNSELECTED ALTERNATIVES:
{alternative_manifest}
"""
    review = llm.structured(QC_SYSTEM, input_text, QualityControlResponse)
    safe_json_dump(review, output_dir / "quality_control.json")
    return review


def apply_qc_changes(
    selected: list[SelectedBlock],
    candidates: list[SelectedBlock],
    review: QualityControlResponse,
    target_words: int,
    chapter_max_share: float = 0.08,
) -> list[SelectedBlock]:
    candidate_map = {b.block_id: b for b in candidates}
    selected_by_chapter: dict[str, list[SelectedBlock]] = {}
    for block in selected:
        selected_by_chapter.setdefault(block.chapter_id, []).append(block)

    remove_ids: set[str] = set()
    for block_id in review.remove_block_ids:
        block = candidate_map.get(block_id)
        if block is None:
            continue
        # Never remove the only retained block for a represented chapter in the reader edition.
        if len(selected_by_chapter.get(block.chapter_id, [])) <= 1:
            continue
        remove_ids.add(block_id)

    revised = [b for b in selected if b.block_id not in remove_ids]
    current_ids = {b.block_id for b in revised}
    words_by_chapter: dict[str, int] = {}
    for block in revised:
        words_by_chapter[block.chapter_id] = words_by_chapter.get(block.chapter_id, 0) + block.word_count

    chapter_cap = max(600, int(target_words * chapter_max_share * 1.15))
    for block_id in review.add_block_ids:
        block = candidate_map.get(block_id)
        if block is None or block_id in current_ids:
            continue
        new_words = sum(b.word_count for b in revised) + block.word_count
        new_chapter_words = words_by_chapter.get(block.chapter_id, 0) + block.word_count
        if new_words <= int(target_words * 1.03) and new_chapter_words <= chapter_cap:
            revised.append(block)
            current_ids.add(block_id)
            words_by_chapter[block.chapter_id] = new_chapter_words
    return revised


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def paragraph_range(block: SelectedBlock, book: Book) -> str:
    first = book.paragraphs[block.paragraph_ids[0]]
    last = book.paragraphs[block.paragraph_ids[-1]]
    page_marker = ""
    if first.page is not None:
        page_marker = f", p. {first.page}" if last.page == first.page else f", pp. {first.page}-{last.page}"
    return f"{block.chapter_id}, {block.paragraph_ids[0]}-{block.paragraph_ids[-1]}{page_marker}"


def assemble_reading_markdown(book: Book, selected: list[SelectedBlock], output_dir: Path) -> Path:
    """Create the actual condensed reading text: almost entirely the author's prose."""
    by_chapter: dict[str, list[SelectedBlock]] = {}
    for block in selected:
        by_chapter.setdefault(block.chapter_id, []).append(block)

    lines: list[str] = [
        f"# {book.title}",
        "",
        "## Condensed reading edition",
        "",
        "*Selected verbatim passages from the original work. Omissions are indicated by three centered dots.*",
        "",
    ]
    for chapter in book.chapters:
        blocks = by_chapter.get(chapter.chapter_id, [])
        if not blocks:
            continue
        lines.extend([f"## {chapter.title}", ""])
        for index, block in enumerate(blocks):
            if index:
                lines.extend(["", "* * *", ""])
            lines.append(block.text.strip())
            lines.append("")
    path = output_dir / "reading_abridgement.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def assemble_audit_markdown(
    book: Book,
    overview: StructuralOverview,
    selected: list[SelectedBlock],
    target_words: int,
    review: QualityControlResponse,
    output_dir: Path,
) -> Path:
    selected_words = sum(b.word_count for b in selected)
    lines = [
        f"# Selection audit: {book.title}", "",
        f"- Source words analysed: {book.total_words:,}",
        f"- Target words: {target_words:,}",
        f"- Selected verbatim words: {selected_words:,}",
        f"- Retained proportion: {selected_words / max(book.total_words, 1):.1%}", "",
        "## Editorial overview", "", overview.overview, "",
        f"**Central question:** {overview.central_question}", "",
        f"**Governing thesis:** {overview.governing_thesis}", "",
        f"**Nonfiction form:** {overview.nonfiction_form}", "",
        "## Selection balance by chapter", "",
        "| Chapter | Source words | Retained words | Retained share of chapter | Share of abridgement | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    retained_by_chapter: dict[str, int] = {}
    for block in selected:
        retained_by_chapter[block.chapter_id] = retained_by_chapter.get(block.chapter_id, 0) + block.word_count
    for chapter in book.chapters:
        if chapter.kind == "exclude":
            continue
        retained = retained_by_chapter.get(chapter.chapter_id, 0)
        status = "represented" if retained else "omitted"
        escaped_title = chapter.title.replace("|", "\\|")
        lines.append(
            f"| {escaped_title} | {chapter.word_count:,} | {retained:,} | "
            f"{retained / max(chapter.word_count, 1):.1%} | {retained / max(selected_words, 1):.1%} | {status} |"
        )
    lines.extend(["", "## Selected passages", ""])
    for block in selected:
        lines.extend([
            f"### {block.block_id}: {block.chapter_title}", "",
            f"- Location: {paragraph_range(block, book)}", f"- Words: {block.word_count}",
            f"- Score: {block.score:.1f}", f"- Function: {block.block_function}", f"- Protected coverage anchor: {block.protected_anchor}", f"- Reason: {block.selection_reason}", "",
        ])
    if review.transition_notes:
        lines.extend(["## Continuity warnings from quality control", ""])
        for note in review.transition_notes:
            lines.append(f"- Before {note.before_block_id}: {note.note}")
    path = output_dir / "selection_audit.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path



def _reader_page_size(name: str) -> tuple[float, float]:
    from reportlab.lib.units import inch
    presets = {
        "small-tablet": (7.0 * inch, 10.0 * inch),
        "a5": (5.827 * inch, 8.268 * inch),
        "large-tablet": (7.5 * inch, 10.5 * inch),
    }
    return presets[name]


def _register_reader_fonts(preference: str = "auto") -> dict[str, str]:
    """
    Embed a locally installed serif family when available.
    No font files are bundled with or distributed in this package.
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    fallback = {
        "regular": "Times-Roman",
        "bold": "Times-Bold",
        "italic": "Times-Italic",
        "bolditalic": "Times-BoldItalic",
        "label": "Times",
    }
    families: list[tuple[str, dict[str, list[str]]]] = [
        (
            "Georgia",
            {
                "regular": [
                    "/System/Library/Fonts/Supplemental/Georgia.ttf",
                    "/Library/Fonts/Georgia.ttf",
                    r"C:\Windows\Fonts\georgia.ttf",
                ],
                "bold": [
                    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
                    "/Library/Fonts/Georgia Bold.ttf",
                    r"C:\Windows\Fonts\georgiab.ttf",
                ],
                "italic": [
                    "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
                    "/Library/Fonts/Georgia Italic.ttf",
                    r"C:\Windows\Fonts\georgiai.ttf",
                ],
                "bolditalic": [
                    "/System/Library/Fonts/Supplemental/Georgia Bold Italic.ttf",
                    "/Library/Fonts/Georgia Bold Italic.ttf",
                    r"C:\Windows\Fonts\georgiaz.ttf",
                ],
            },
        ),
        (
            "DejaVu Serif",
            {
                "regular": [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
                    "/Library/Fonts/DejaVuSerif.ttf",
                ],
                "bold": [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
                    "/Library/Fonts/DejaVuSerif-Bold.ttf",
                ],
                "italic": [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
                    "/Library/Fonts/DejaVuSerif-Italic.ttf",
                ],
                "bolditalic": [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf",
                    "/Library/Fonts/DejaVuSerif-BoldItalic.ttf",
                ],
            },
        ),
    ]
    requested = preference.strip().lower()
    if requested == "times":
        return fallback

    for family_name, candidates in families:
        family_keys = {family_name.lower(), family_name.lower().replace(" ", "")}
        if requested != "auto" and requested not in family_keys:
            continue
        resolved: dict[str, str] = {}
        for style, locations in candidates.items():
            selected_path = next((loc for loc in locations if Path(loc).exists()), None)
            if not selected_path:
                resolved = {}
                break
            resolved[style] = selected_path
        if not resolved:
            continue
        try:
            prefix = family_name.replace(" ", "") + "_Reader"
            registered: dict[str, str] = {}
            for style, selected_path in resolved.items():
                registered_name = f"{prefix}_{style}"
                pdfmetrics.registerFont(TTFont(registered_name, selected_path))
                registered[style] = registered_name
            registered["label"] = family_name
            return registered
        except Exception as exc:
            LOGGER.warning("Unable to embed %s in PDF output: %s. Using built-in Times.", family_name, exc)
    return fallback


def _pdf_safe_text(value: str) -> str:
    from xml.sax.saxutils import escape
    cleaned = normalize_text(value).replace("\u2028", " ").replace("\u2029", " ")
    return escape(cleaned)


def export_reading_pdf(
    book: Book,
    selected: list[SelectedBlock],
    output_path: Path,
    page_preset: str = "small-tablet",
    body_font_size: float = 14.0,
    font_preference: str = "auto",
) -> None:
    """Render a large-type tablet reading edition directly from selected source blocks."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.platypus import (
            BaseDocTemplate,
            Flowable,
            Frame,
            PageBreak,
            PageTemplate,
            Paragraph,
            Spacer,
        )
    except ImportError as exc:
        raise ImportError(
            "PDF output requires ReportLab. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    page_size = _reader_page_size(page_preset)
    fonts = _register_reader_fonts(font_preference)
    selected_words = sum(block.word_count for block in selected)

    class OmissionRule(Flowable):
        def __init__(self, width: float = 72.0) -> None:
            super().__init__()
            self.width = width
            self.height = 26.0

        def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
            self.available_width = available_width
            return available_width, self.height

        def draw(self) -> None:
            from reportlab.lib import colors
            x = (self.available_width - self.width) / 2
            y = self.height / 2
            self.canv.setStrokeColor(colors.HexColor("#D0C8BC"))
            self.canv.setLineWidth(0.55)
            self.canv.line(x, y, x + 22, y)
            self.canv.line(x + self.width - 22, y, x + self.width, y)
            self.canv.setFillColor(colors.HexColor("#82786D"))
            self.canv.setFont(fonts["regular"], 9)
            self.canv.drawCentredString(x + self.width / 2, y - 3.2, "\u2022 \u2022 \u2022")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReaderTitle", parent=styles["Title"], fontName=fonts["bold"], fontSize=27,
        leading=34, alignment=TA_CENTER, textColor=colors.HexColor("#24201C"), spaceAfter=12,
    )
    subtitle_style = ParagraphStyle(
        "ReaderSubtitle", parent=styles["Normal"], fontName=fonts["italic"], fontSize=13,
        leading=20, alignment=TA_CENTER, textColor=colors.HexColor("#60574D"), spaceAfter=8,
    )
    note_style = ParagraphStyle(
        "ReaderNote", parent=styles["Normal"], fontName=fonts["regular"], fontSize=10,
        leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#756C63"),
    )
    chapter_style = ParagraphStyle(
        "ReaderChapter", parent=styles["Heading1"], fontName=fonts["bold"], fontSize=21,
        leading=29, alignment=TA_LEFT, textColor=colors.HexColor("#28231F"), spaceAfter=25,
        keepWithNext=False,
    )
    body_style = ParagraphStyle(
        "ReaderBody", parent=styles["BodyText"], fontName=fonts["regular"], fontSize=body_font_size,
        leading=body_font_size * 1.58, alignment=TA_LEFT, textColor=colors.HexColor("#211E1A"),
        spaceAfter=body_font_size * 0.66, firstLineIndent=body_font_size * 1.42,
        splitLongWords=False, allowWidows=0, allowOrphans=0,
    )
    first_body_style = ParagraphStyle("ReaderFirstBody", parent=body_style, firstLineIndent=0)

    left_margin = 0.67 * inch
    right_margin = 0.67 * inch
    top_margin = 0.72 * inch
    bottom_margin = 0.66 * inch
    frame = Frame(
        left_margin, bottom_margin,
        page_size[0] - left_margin - right_margin,
        page_size[1] - top_margin - bottom_margin,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="reader-frame",
    )
    doc = BaseDocTemplate(
        str(output_path), pagesize=page_size,
        title=f"{book.title} - Condensed Reading Edition",
        leftMargin=left_margin, rightMargin=right_margin, topMargin=top_margin, bottomMargin=bottom_margin,
    )

    def page_decor(canvas: Any, document: Any) -> None:
        page = canvas.getPageNumber()
        if page == 1:
            return
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#756C62"))
        canvas.setStrokeColor(colors.HexColor("#D8D1C7"))
        canvas.setLineWidth(0.45)
        canvas.setFont(fonts["regular"], 8.3)
        header = book.title.upper()
        max_width = page_size[0] - left_margin - right_margin - 16
        while stringWidth(header, fonts["regular"], 8.3) > max_width and len(header) > 20:
            header = header[:-2].rstrip() + "\u2026"
        canvas.drawString(left_margin, page_size[1] - 0.40 * inch, header)
        canvas.line(left_margin, page_size[1] - 0.50 * inch, page_size[0] - right_margin, page_size[1] - 0.50 * inch)
        canvas.setFont(fonts["regular"], 9)
        canvas.drawCentredString(page_size[0] / 2, 0.38 * inch, str(page))
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="reader", frames=[frame], onPage=page_decor)])

    by_chapter: dict[str, list[SelectedBlock]] = {}
    for block in selected:
        by_chapter.setdefault(block.chapter_id, []).append(block)

    story: list[Any] = [
        Spacer(1, 1.44 * inch),
        Paragraph(_pdf_safe_text(book.title), title_style),
        Spacer(1, 0.10 * inch),
        Paragraph("Condensed Reading Edition", subtitle_style),
        Spacer(1, 0.40 * inch),
        Paragraph(
            f"Selected verbatim passages from the original work. "
            f"Omissions are marked discreetly. Retained text: {selected_words:,} words.",
            note_style,
        ),
        PageBreak(),
    ]

    first_section = True
    for chapter in book.chapters:
        blocks = by_chapter.get(chapter.chapter_id, [])
        if not blocks:
            continue
        if not first_section:
            story.append(PageBreak())
        first_section = False
        story.extend([Spacer(1, 0.20 * inch), Paragraph(_pdf_safe_text(chapter.title), chapter_style)])
        first_para = True
        for block_index, block in enumerate(blocks):
            if block_index:
                story.extend([Spacer(1, 4), OmissionRule(), Spacer(1, 8)])
                first_para = True
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", block.text.strip()) if p.strip()]
            for paragraph in paragraphs:
                story.append(Paragraph(_pdf_safe_text(paragraph), first_body_style if first_para else body_style))
                first_para = False

    doc.build(story)
    LOGGER.info(
        "Generated reading PDF: %s (%s, %.1f pt body type, %s).",
        output_path, page_preset, body_font_size, fonts["label"],
    )


def export_reading_docx(markdown_path: Path, output_path: Path) -> None:
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Inches, Pt
    except ImportError:
        LOGGER.warning("python-docx is not installed; skipping DOCX export.")
        return

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)
    normal = doc.styles["Normal"]
    normal.font.name = "Georgia"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.08

    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=0)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=1)
        elif line.strip() == "* * *":
            para = doc.add_paragraph("* * *")
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.strip().startswith("*") and line.strip().endswith("*"):
            para = doc.add_paragraph(line.strip().strip("*"))
            para.italic = True
        elif line.strip():
            doc.add_paragraph(line.strip())
    doc.save(str(output_path))


# ---------------------------------------------------------------------------
# Persistence and execution
# ---------------------------------------------------------------------------

def persist_parsed_book(book: Book, output_dir: Path) -> None:
    metadata = {
        "title": book.title,
        "source_path": book.source_path,
        "total_words": book.total_words,
        "chapters": [asdict(c) for c in book.chapters],
        "diagnostics": book.diagnostics,
    }
    safe_json_dump(metadata, output_dir / "book_metadata.json")
    with (output_dir / "book_paragraphs.jsonl").open("w", encoding="utf-8") as fh:
        for chapter in book.chapters:
            for pid in chapter.paragraph_ids:
                fh.write(json.dumps(asdict(book.paragraphs[pid]), ensure_ascii=False) + "\n")




def write_structure_report(book: Book, output_dir: Path) -> Path:
    excluded_words = sum(c.word_count for c in book.chapters if c.kind == "exclude")
    diagnostics = book.diagnostics or {}
    warnings = diagnostics.get("warnings", [])
    lines = [
        f"# Parsed structure report: {book.title}", "",
        "## Parser confidence", "",
        f"- Format: {diagnostics.get('format', Path(book.source_path).suffix.upper().lstrip('.'))}",
        f"- Navigation source: {diagnostics.get('navigation_source', 'not reported')}",
        f"- Parser confidence: {diagnostics.get('parser_confidence', 'not reported')}",
        f"- Proceed with LLM extraction: {'yes' if diagnostics.get('proceed_with_llm_extraction', True) else 'no'}",
        f"- Fixed-layout indicators: {', '.join(diagnostics.get('fixed_layout_indicators', [])) or 'none detected'}",
        f"- Image-only content risk: {diagnostics.get('image_only_content_risk', 'not assessed')}",
        f"- Unassigned text words: {diagnostics.get('unassigned_text_words', 0):,}",
        f"- Prose fallback items: {diagnostics.get('prose_fallback_items', 0):,}",
        "",
        "## Recovered structure", "",
        f"- Included source words: {book.total_words:,}",
        f"- Excluded/non-reading words: {excluded_words:,}",
        f"- Sections detected: {len(book.chapters)}", "",
        "| ID | Type | Section title | Words |",
        "|---|---|---|---:|",
    ]
    for chapter in book.chapters:
        safe_title = chapter.title.replace("|", "\\|")
        lines.append(f"| {chapter.chapter_id} | {chapter.kind} | {safe_title} | {chapter.word_count:,} |")
    if warnings:
        lines.extend(["", "## Parser warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])
    lines.extend([
        "",
        "Review this structure before incurring API calls when parser confidence is medium or low. "
        "The program automatically stops before API calls when recovery is insufficient.",
        "",
    ])
    path = output_dir / "parsed_structure_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def reset_generated_outputs(output_dir: Path) -> None:
    """Remove prior generated artefacts while leaving unrelated user files untouched."""
    generated_files = [
        "book_metadata.json", "book_paragraphs.jsonl", "parsed_structure_report.md",
        "structural_overview.json", "scored_candidates.json", "global_selection.json",
        "quality_control.json", "selection_audit.md", "reading_abridgement.md",
        "reading_abridgement.docx", "reading_abridgement.pdf",
    ]
    for filename in generated_files:
        target = output_dir / filename
        if target.exists():
            target.unlink()
    candidates_dir = output_dir / "chapter_candidates"
    if candidates_dir.exists():
        shutil.rmtree(candidates_dir)


def run_pipeline(args: argparse.Namespace) -> Path:
    source = Path(args.input_file).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reset_generated_outputs(output_dir)

    LOGGER.info("Loading source book: %s", source)
    book = load_book(source, chapter_map_path=Path(args.chapter_map).expanduser().resolve() if args.chapter_map else None)
    persist_parsed_book(book, output_dir)
    structure_report = write_structure_report(book, output_dir)
    validate_book_structure(book)
    LOGGER.info("Parsed %d chapters/sections and %s included words.", len(book.chapters), f"{book.total_words:,}")
    if args.parse_only:
        LOGGER.info("Parse-only mode complete: %s", structure_report)
        return structure_report

    llm = LLM(model=args.model, retries=args.retries)
    overview = create_structural_overview(
        llm=llm,
        book=book,
        emphasis=args.emphasis,
        max_structural_words=args.max_structural_words,
        output_dir=output_dir,
    )

    candidates = all_candidate_blocks(
        llm=llm,
        book=book,
        overview=overview,
        emphasis=args.emphasis,
        candidate_ratio=args.candidate_ratio,
        chapter_chunk_words=args.chapter_chunk_words,
        output_dir=output_dir,
    )
    if not candidates:
        raise RuntimeError("No candidate blocks were selected. Review source parsing and model output.")

    scored = score_candidates(
        llm=llm,
        candidates=candidates,
        overview=overview,
        score_batch_size=args.score_batch_size,
        output_dir=output_dir,
    )

    selected, target_words = choose_blocks_under_budget(
        book=book,
        candidates=scored,
        overview=overview,
        target_ratio=args.target_ratio,
        coverage_mode=args.coverage_mode,
        chapter_max_share=args.chapter_max_share,
    )
    LOGGER.info(
        "Initial global selection retains %s words against a target of %s.",
        f"{sum(b.word_count for b in selected):,}",
        f"{target_words:,}",
    )

    review = quality_control(
        llm=llm,
        overview=overview,
        selected=selected,
        candidates=scored,
        target_words=target_words,
        output_dir=output_dir,
    )
    if args.apply_qc:
        selected = apply_qc_changes(selected, scored, review, target_words, chapter_max_share=args.chapter_max_share)

    global_payload = {
        "target_ratio": args.target_ratio,
        "target_words": target_words,
        "selected_words": sum(b.word_count for b in selected),
        "selected_ratio": sum(b.word_count for b in selected) / max(book.total_words, 1),
        "selected_blocks": [asdict(b) for b in selected],
        "quality_control_applied": bool(args.apply_qc),
        "nonfiction_form": overview.nonfiction_form,
        "coverage_mode": args.coverage_mode,
        "chapter_max_share": args.chapter_max_share,
    }
    safe_json_dump(global_payload, output_dir / "global_selection.json")

    reading_path = assemble_reading_markdown(book=book, selected=selected, output_dir=output_dir)
    assemble_audit_markdown(book=book, overview=overview, selected=selected, target_words=target_words, review=review, output_dir=output_dir)
    pdf_path = output_dir / "reading_abridgement.pdf"
    export_reading_pdf(
        book=book,
        selected=selected,
        output_path=pdf_path,
        page_preset=args.pdf_page_size,
        body_font_size=args.pdf_font_size,
        font_preference=args.pdf_font,
    )
    if not args.no_docx:
        export_reading_docx(reading_path, output_dir / "reading_abridgement.docx")

    LOGGER.info("Completed readable extractive abridgement PDF: %s", pdf_path)
    return pdf_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a rights-cleared, quotation-dominant extractive abridgement of a nonfiction book using the OpenAI API."
    )
    parser.add_argument("input_file", help="Input EPUB, PDF, DOCX, TXT, or Markdown book file.")
    parser.add_argument("--output-dir", default="abridgement_output", help="Directory for final and intermediate outputs.")
    parser.add_argument("--chapter-map", help="Optional JSON chapter map for PDFs without reliable embedded bookmarks. Pages are 1-indexed.")
    parser.add_argument("--parse-only", action="store_true", help="Parse and clean the book, write a structure report, and stop before API calls.")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        help="OpenAI model supporting structured outputs. Defaults to OPENAI_MODEL or gpt-5-mini.",
    )
    parser.add_argument("--target-ratio", type=float, default=0.25, help="Final retained source-word ratio. Default: 0.25.")
    parser.add_argument(
        "--coverage-mode",
        choices=["all", "major", "none"],
        default="all",
        help="Chapter coverage rule. 'all' retains at least one passage per parsed non-back-matter section; default: all.",
    )
    parser.add_argument(
        "--chapter-max-share",
        type=float,
        default=0.08,
        help="Maximum share of final words assigned to one chapter before high-priority tolerance; default: 0.08.",
    )
    parser.add_argument(
        "--candidate-ratio",
        type=float,
        default=0.42,
        help="Candidate quotation share requested per chapter before global pruning. Default: 0.42.",
    )
    parser.add_argument(
        "--emphasis",
        default="Preserve the governing thesis, major claims, essential definitions and concepts, causal or technical mechanisms, decisive evidence, necessary case material, important counterarguments or limitations, and conclusions or implications.",
        help="Selection emphasis supplied to the model.",
    )
    parser.add_argument(
        "--chapter-chunk-words",
        type=int,
        default=18000,
        help="Split unusually long chapters above this word count for selection calls. Default: 18000.",
    )
    parser.add_argument(
        "--max-structural-words",
        type=int,
        default=24000,
        help="Maximum introduction/ending words supplied to the overview call. Default: 24000.",
    )
    parser.add_argument("--score-batch-size", type=int, default=20, help="Candidate blocks per scoring call. Default: 20.")
    parser.add_argument("--retries", type=int, default=3, help="Maximum retries per API call. Default: 3.")
    parser.add_argument(
        "--apply-qc",
        action="store_true",
        help="Apply final quality-control add/remove recommendations when they remain within the word budget tolerance.",
    )
    parser.add_argument(
        "--pdf-page-size",
        choices=["small-tablet", "a5", "large-tablet"],
        default="small-tablet",
        help="PDF page preset. Default: small-tablet (7 x 10 inches).",
    )
    parser.add_argument(
        "--pdf-font-size",
        type=float,
        default=14.0,
        help="Body type size in points for the PDF reading edition. Default: 14.0.",
    )
    parser.add_argument(
        "--pdf-font",
        choices=["auto", "georgia", "dejavu serif", "dejavuserif", "times"],
        default="auto",
        help="Serif font for the PDF. Auto embeds Georgia or DejaVu Serif when available; otherwise uses Times.",
    )
    parser.add_argument("--no-docx", action="store_true", help="Do not generate an optional DOCX copy of the abridgement.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.target_ratio < 1:
        raise ValueError("--target-ratio must be between 0 and 1.")
    if not args.target_ratio < args.candidate_ratio <= 1:
        raise ValueError("--candidate-ratio must be greater than --target-ratio and no greater than 1.")
    if not 0.01 <= args.chapter_max_share <= 0.50:
        raise ValueError("--chapter-max-share must be between 0.01 and 0.50.")
    if args.chapter_chunk_words < 1000:
        raise ValueError("--chapter-chunk-words must be at least 1000.")
    if args.score_batch_size < 1:
        raise ValueError("--score-batch-size must be positive.")
    if not 11.0 <= args.pdf_font_size <= 20.0:
        raise ValueError("--pdf-font-size must be between 11 and 20 points.")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        validate_args(args)
        output = run_pipeline(args)
    except Exception as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1
    print(f"\nReadable abridgement written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
