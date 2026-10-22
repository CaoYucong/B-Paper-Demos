"""Language switch. Widget values are not translated."""

from src.ui.i18n import L, plain, present_text, shown_body, shown_label, t


def test_both_languages_stay_in_the_page():
    html = t("Section", "章节")
    assert 'class="b14-en"' in html and "Section" in html
    assert 'class="b14-zh"' in html and "章节" in html
    assert L("Open / 进入") == "Open / 进入"
    assert shown_label("迭代次数") == "Iterations / 迭代次数"
    assert shown_label(html) == "Section / 章节"
    assert plain("Section", "章节") == "Section / 章节"


def test_numbers_after_the_first_slash_stay_with_their_language():
    html = t("Score: 3 / 4", "得分：3 / 4")
    assert "Score: 3 / 4" in html
    assert "得分：3 / 4" in html


def test_a_short_caption_stays_available_for_every_language():
    chinese = shown_body("抽样均方误差 0.0609。量化均方误差 0。")
    assert "b14-only" in chinese and "b14-zh" in chinese and "抽样均方误差" in chinese
    english = shown_body("Sampling MSE 0.0609. Quantisation MSE 0. The sampled array is 32 by 32 before display.")
    assert "b14-only" in english and "b14-en" in english and "Sampling MSE" in english


def test_section_anchors_are_not_marked_as_one_language():
    anchor = "<div id='Medianfiltering'></div>"
    assert shown_body(anchor) == anchor


def test_formula_blocks_are_kept():
    formula = "$$\ng = Af + n\n$$"
    assert shown_body(formula) == formula


def test_sidebar_links_and_bold_labels_keep_both_languages():
    link, link_html = present_text(f"&nbsp;&nbsp;[{t('Median filtering', '中值滤波')}](#Medianfiltering)")
    assert link_html
    assert 'href="#Medianfiltering"' in link
    assert "中值滤波" in link
    bold, bold_html = present_text(f"**{t('Original', '原图')}**")
    assert bold_html and bold.startswith("<strong>") and "原图" in bold
    titled, titled_html = present_text(f"B14 · {t('Image Processing', '图像处理')}")
    assert titled_html and "B14 · " in titled and "图像处理" in titled
