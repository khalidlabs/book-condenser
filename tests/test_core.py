import argparse
import json
from pathlib import Path

import pytest

from book_condenser.core import (
    Book,
    Chapter,
    Paragraph,
    QualityControlResponse,
    SelectedBlock,
    StructuralOverview,
    apply_qc_changes,
    choose_blocks_under_budget,
    classify_chapter,
    load_manual_chapter_map,
    normalize_text,
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
        book, candidates, minimal_overview(), target_ratio=0.25, coverage_mode="all"
    )

    assert target_words == 500
    assert {item.chapter_id for item in selected} == {"CH001", "CH002"}
    assert all(item.protected_anchor for item in selected)


def test_apply_qc_keeps_only_block_for_represented_chapter() -> None:
    selected = [block("B1", "CH001", 100, 10.0, "P1"), block("B2", "CH002", 100, 9.0, "P2")]
    review = QualityControlResponse(
        assessment="test",
        missing_coverage=[],
        remove_block_ids=["B1"],
        add_block_ids=[],
        transition_notes=[],
    )

    revised = apply_qc_changes(selected, selected, review, target_words=500)

    assert [item.block_id for item in revised] == ["B1", "B2"]


def test_validate_args_rejects_invalid_ratios() -> None:
    args = argparse.Namespace(
        target_ratio=0.5,
        candidate_ratio=0.4,
        chapter_max_share=0.08,
        chapter_chunk_words=18000,
        score_batch_size=20,
        pdf_font_size=14.0,
    )

    with pytest.raises(ValueError, match="candidate-ratio"):
        validate_args(args)

