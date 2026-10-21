"""Catalog and practice-answer checks."""

from __future__ import annotations

from src.catalog import PAPERS, paper_by_id
from src.practice.problems import PROBLEMS, QUIZ, grade_numeric


def test_catalog_keeps_extensions_as_placeholders():
    codes = [paper.code for paper in PAPERS]
    assert codes == ["B14", "B15", "B16"]
    b14 = paper_by_id("b14")
    image = next(module for module in b14.modules if module.id == "image-processing")
    signal = next(module for module in b14.modules if module.id == "signal-processing")
    assert image.status == "available"
    assert signal.status == "available"
    assert signal.title_en == "Signal Analysis"
    assert [section.id for section in image.sections] == ["home", "l1", "l2", "l3", "l4", "practice"]
    assert [section.id for section in signal.sections] == ["home", "l1", "l2", "l3", "l4", "practice"]
    assert paper_by_id("b15").status == "planned"


def test_practice_answers_match_the_grader():
    assert len(PROBLEMS) == 8
    assert len(QUIZ) == 10
    for problem in PROBLEMS:
        assert grade_numeric(problem.id, problem.answer)
        assert not grade_numeric(problem.id, problem.answer + problem.tolerance + 1.0)
    assert len({item.correct for item in QUIZ}) >= 1
    assert all(0 <= item.correct < len(item.choices_en) for item in QUIZ)
