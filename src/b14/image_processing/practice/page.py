"""Original exercises and a short quiz. Not the official example sheet."""

from __future__ import annotations

import streamlit as st

from src.b14.image_processing.practice.problems import PROBLEMS, QUIZ, grade_numeric
from src.b14.image_processing.experiments import PRACTICE_LINKS
from src.ui.i18n import say, t


def render_practice() -> None:
    st.markdown("<div id='practice'></div>", unsafe_allow_html=True)
    st.title(t("Practice", "练习"))
    st.warning(
        t(
            "Original practice problems. The example-sheet PDF was found, but no question text could be extracted from it, "
            "so these prompts are not the official B14 example sheet and do not copy its wording.",
            "原创练习。找到了例题 PDF，但没有从中提取出题目文字，因此这些题不是 B14 官方例题，也没有抄写例题原文。",
        )
    )
    for problem in PROBLEMS:
        with st.container(border=True):
            st.markdown(f"**{t(problem.topic_en, problem.topic_zh)}**")
            say(problem.prompt_en, problem.prompt_zh)
            submitted = st.number_input(
                "Your answer / 你的答案",
                value=0.0,
                step=0.1,
                format="%.6f",
                key=f"answer-{problem.id}",
            )
            if st.button("Check / 检查", key=f"check-{problem.id}"):
                st.session_state[f"mark-{problem.id}"] = grade_numeric(problem.id, submitted)
            mark = st.session_state.get(f"mark-{problem.id}")
            if mark is True:
                st.success("Within tolerance. / 在容差以内。")
            elif mark is False:
                st.error("Outside tolerance. / 超出容差。")
            with st.expander("Worked solution / 题解"):
                say(problem.solution_en, problem.solution_zh)
                st.write(t(f"Answer: {problem.answer:.6g}", f"答案：{problem.answer:.6g}"))
            anchor = PRACTICE_LINKS[problem.id]
            st.markdown(f"[{t('Go to this experiment', '滚到这个实验')}](#{anchor})")

    st.header(t("Quiz", "小测"))
    st.caption(
        t(
            "Covers all four lectures, including restoration topics that are not in the numeric problems above.",
            "覆盖四讲，包括上面数值题里没有的复原内容。",
        )
    )
    for question in QUIZ:
        labels = [f"{en} / {zh}" for en, zh in zip(question.choices_en, question.choices_zh)]
        st.radio(t(question.prompt_en, question.prompt_zh), labels, key=f"quiz-{question.id}")
    if st.button("Score the quiz / 给小测评分", type="primary"):
        correct = 0
        details = []
        for question in QUIZ:
            labels = [f"{en} / {zh}" for en, zh in zip(question.choices_en, question.choices_zh)]
            selected = st.session_state[f"quiz-{question.id}"]
            is_correct = selected == labels[question.correct]
            correct += int(is_correct)
            details.append((question, is_correct))
        st.session_state["quiz_details"] = details
        st.session_state["quiz_score"] = (correct, len(QUIZ))
    if "quiz_score" in st.session_state:
        correct, total = st.session_state["quiz_score"]
        st.write(t(f"Score: {correct} / {total}", f"得分：{correct} / {total}"))
        for question, is_correct in st.session_state.get("quiz_details", []):
            if is_correct:
                st.markdown("- " + t(question.prompt_en, question.prompt_zh))
                st.caption(t("Correct. " + question.explanation_en, "正确。" + question.explanation_zh))
            else:
                st.markdown("- " + t(question.prompt_en, question.prompt_zh))
                st.caption(t(question.explanation_en, question.explanation_zh))
