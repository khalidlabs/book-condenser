import argparse
import json
from datetime import datetime
from pathlib import Path

import pytest

from book_condenser.core import (
    AnalyticalCoverageAssessment,
    AnalyticalMap,
    Book,
    Chapter,
    Paragraph,
    QualityControlResponse,
    SelectedBlock,
    StructuralOverview,
    allocate_output_dir,
    apply_qc_changes,
    chapter_word_caps,
    choose_blocks_under_budget,
    classify_chapter,
    load_manual_chapter_map,
    normalize_text,
    output_dir_slug,
    paragraph_split,
    validate_args,
)


def minimal_overview() -> StructuralOverview:
    return StructuralOverview(
        nonfiction_form="argumentative",
        central_question="What is being argued?",
        governing_thesis="A test thesis.",
        overview="A short overview.",
        chronological_or_logical_arc=[],
        core_claims=[],
        key_concepts=[],
        evidence_and_cases=[],
        counterarguments_or_limitations=[],
        chapter_priorities=[],
        selection_rules=[],
    )


def minimal_analytical_map() -> AnalyticalMap:
    return AnalyticalMap(
        book_classification="test",
        central_problem="test",
        unity_statement="test",
        requirements=[],
        argument_relations=[],
        minimum_complete_reading_path=[],
    )


def minimal_qc_review() -> QualityControlResponse:
    return QualityControlResponse(
        assessment="test",
        analytical_coverage=AnalyticalCoverageAssessment(
            unity_preserved=True,
            essential_terms_preserved=True,
            major_propositions_preserved=True,
            supporting_arguments_preserved=True,
            objections_or_limitations_preserved=True,
            conclusion_preserved=True,
            missing_requirement_ids=[],
            unsupported_proposition_ids=[],
            undefined_term_ids=[],
        ),
        missing_coverage=[],
        remove_block_ids=[],
        add_block_ids=[],
        transition_notes=[],
    )


def block(block_id: str, chapter_id: str, words: int, score: float, paragraph_id: str) -> SelectedBlock:
    return SelectedBlock(
        block_id=block_id,
        chapter_id=chapter_id,
        chapter_title=chapter_id,
        paragraph_ids=[paragraph_id],
        word_count=words,
        selection_reason="test",
        importance="high",
        themes=["test"],
        redundancy_risk="low",
        block_function="claim",
        text="word " * words,
        score=score,
    )


def test_normalize_text_repairs_known_pdf_artifacts() -> None:
    assert normalize_text("T he plan was brain\ufffewounded and EG&GDr. Smith") == (
        "The plan was brain-wounded and EG&G Dr. Smith"
    )


def test_paragraph_split_discards_tiny_blocks() -> None:
    assert paragraph_split("A\n\nA meaningful paragraph remains.") == ["A meaningful paragraph remains."]


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Introduction", "front"),
        ("Chapter One", "body"),
        ("Conclusion", "ending"),
        ("Bibliography", "exclude"),
        ("List of Interviews and Written Correspondence", "exclude"),
    ],
)
def test_classify_chapter(title: str, expected: str) -> None:
    assert classify_chapter(title) == expected


def test_load_manual_chapter_map_validates_required_fields(tmp_path: Path) -> None:
    chapter_map = tmp_path / "chapters.json"
    chapter_map.write_text(json.dumps([{"title": "Chapter One", "start_page": 1}]), encoding="utf-8")

    assert load_manual_chapter_map(chapter_map) == [{"title": "Chapter One", "start_page": 1}]

    chapter_map.write_text(json.dumps([{"title": "Missing start"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="title and start_page"):
        load_manual_chapter_map(chapter_map)


def test_choose_blocks_protects_chapter_coverage() -> None:
    paragraphs = {
        "P1": Paragraph("P1", "CH001", "One", 0, "one", 1),
        "P2": Paragraph("P2", "CH002", "Two", 1, "two", 1),
    }
    book = Book(
        title="Synthetic",
        source_path="synthetic.txt",
        chapters=[
            Chapter("CH001", "One", ["P1"], 1000),
            Chapter("CH002", "Two", ["P2"], 1000),
        ],
        paragraphs=paragraphs,
        total_words=2000,
    )
    candidates = [
        block("B1", "CH001", 100, 10.0, "P1"),
        block("B2", "CH002", 100, 1.0, "P2"),
    ]

    selected, target_words = choose_blocks_under_budget(
        book,
        candidates,
        minimal_overview(),
        minimal_analytical_map(),
        target_ratio=0.25,
        coverage_mode="all",
    )

    assert target_words == 500
    assert {item.chapter_id for item in selected} == {"CH001", "CH002"}
    assert all(item.protected_anchor for item in selected)


def test_chapter_word_caps_proportional() -> None:
    book = Book(
        title="Synthetic",
        source_path="synthetic.txt",
        chapters=[
            Chapter("CH001", "Long", [], 8000),
            Chapter("CH002", "Short", [], 2000),
        ],
        paragraphs={},
        total_words=10000,
    )
    caps = chapter_word_caps(book, target_words=2500, priority={})
    assert caps == {"CH001": 2000, "CH002": 600}


def test_chapter_word_caps_high_priority_multiplier() -> None:
    book = Book(
        title="Synthetic",
        source_path="synthetic.txt",
        chapters=[Chapter("CH001", "One", [], 1000)],
        paragraphs={},
        total_words=1000,
    )
    caps = chapter_word_caps(book, target_words=1000, priority={"CH001": "high"})
    assert caps["CH001"] == int(max(600, 1000) * 1.15)


def test_choose_blocks_respects_proportional_chapter_caps() -> None:
    paragraphs = {
        f"P{i}": Paragraph(f"P{i}", "CH001" if i < 8 else "CH002", "Ch", i, "text", 1)
        for i in range(1, 9)
    }
    book = Book(
        title="Synthetic",
        source_path="synthetic.txt",
        chapters=[
            Chapter("CH001", "Long", [f"P{i}" for i in range(1, 8)], 3000),
            Chapter("CH002", "Short", ["P8"], 1000),
        ],
        paragraphs=paragraphs,
        total_words=4000,
    )
    candidates = [
        block(f"B{i}", "CH001", 200, float(20 - i), f"P{i}")
        for i in range(1, 8)
    ] + [block("B8", "CH002", 200, 5.0, "P8")]
    caps = chapter_word_caps(book, target_words=1000, priority={})

    selected, target_words = choose_blocks_under_budget(
        book,
        candidates,
        minimal_overview(),
        minimal_analytical_map(),
        target_ratio=0.25,
        coverage_mode="none",
    )

    assert target_words == 1000
    words_by_chapter: dict[str, int] = {}
    for item in selected:
        words_by_chapter[item.chapter_id] = words_by_chapter.get(item.chapter_id, 0) + item.word_count
    for chapter_id, retained in words_by_chapter.items():
        assert retained <= caps[chapter_id]


def test_apply_qc_keeps_only_block_for_represented_chapter() -> None:
    book = Book(
        title="Synthetic",
        source_path="synthetic.txt",
        chapters=[
            Chapter("CH001", "One", [], 1000),
            Chapter("CH002", "Two", [], 1000),
        ],
        paragraphs={},
        total_words=2000,
    )
    selected = [block("B1", "CH001", 100, 10.0, "P1"), block("B2", "CH002", 100, 9.0, "P2")]
    review = minimal_qc_review()
    review = review.model_copy(update={"remove_block_ids": ["B1"]})

    revised = apply_qc_changes(selected, selected, review, target_words=500, book=book, overview=minimal_overview())

    assert [item.block_id for item in revised] == ["B1", "B2"]


def test_validate_args_rejects_invalid_ratios() -> None:
    args = argparse.Namespace(
        target_ratio=0.5,
        candidate_ratio=0.4,
        chapter_chunk_words=18000,
        score_batch_size=20,
        pdf_font_size=14.0,
    )

    with pytest.raises(ValueError, match="candidate-ratio"):
        validate_args(args)


def test_output_dir_slug_sanitizes_source_names() -> None:
    assert output_dir_slug("Eric Greitens - Resilience") == "Eric-Greitens-Resilience"


def test_allocate_output_dir_creates_unique_run_folder(tmp_path: Path) -> None:
    source = tmp_path / "Eric Greitens - Resilience.epub"
    source.write_text("placeholder", encoding="utf-8")
    fixed_now = datetime(2026, 5, 24, 12, 30, 45)

    first = allocate_output_dir(tmp_path / "out", source, now=fixed_now)
    second = allocate_output_dir(tmp_path / "out", source, now=fixed_now)

    assert first == tmp_path / "out" / "Eric-Greitens-Resilience-20260524-123045"
    assert second == tmp_path / "out" / "Eric-Greitens-Resilience-20260524-123045-2"
    assert first.exists()
    assert second.exists()


def test_allocate_output_dir_reuse_writes_to_exact_folder(tmp_path: Path) -> None:
    source = tmp_path / "book.epub"
    source.write_text("placeholder", encoding="utf-8")
    target = tmp_path / "out" / "fixed-run"

    output_dir = allocate_output_dir(target, source, reuse=True)

    assert output_dir == target.resolve()
    assert output_dir.exists()

