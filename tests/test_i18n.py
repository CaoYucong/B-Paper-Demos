"""Language switch. Widget values are not translated."""

from src.ui.i18n import L, shown_body, shown_label, t


def test_bilingual_is_the_default(monkeypatch):
    monkeypatch.setattr("src.ui.i18n.language", lambda: "both")
    assert t("Section", "章节") == "Section / 章节"
    assert L("Section / 章节") == "Section / 章节"


def test_each_language_keeps_one_side(monkeypatch):
    monkeypatch.setattr("src.ui.i18n.language", lambda: "en")
    assert t("Section", "章节") == "Section"
    assert L("Open / 进入") == "Open"
    assert shown_label("迭代次数") == "Iterations"

    monkeypatch.setattr("src.ui.i18n.language", lambda: "zh")
    assert t("Section", "章节") == "章节"
    assert L("Open / 进入") == "进入"
    assert shown_label("迭代次数") == "迭代次数"


def test_numbers_after_the_first_slash_are_not_split_away(monkeypatch):
    monkeypatch.setattr("src.ui.i18n.language", lambda: "en")
    assert t("Score: 3 / 4", "得分：3 / 4") == "Score: 3 / 4"


def test_a_short_caption_in_the_other_language_is_omitted(monkeypatch):
    monkeypatch.setattr("src.ui.i18n.language", lambda: "en")
    assert shown_body("抽样均方误差 0.0609。量化均方误差 0。") is None
    monkeypatch.setattr("src.ui.i18n.language", lambda: "zh")
    assert shown_body("Sampling MSE 0.0609. Quantisation MSE 0. The sampled array is 32 by 32 before display.") is None


def test_formula_blocks_are_kept(monkeypatch):
    monkeypatch.setattr("src.ui.i18n.language", lambda: "en")
    formula = "$$\ng = Af + n\n$$"
    assert shown_body(formula) == formula
