"""Course catalog for the B-paper demonstration.

Add a paper or a module here. The shell reads this list and does not
hard-code B15, B16, or Signal Processing content.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Lecture:
    id: str
    title_en: str
    title_zh: str


@dataclass(frozen=True)
class LabSection:
    id: str
    title_en: str
    title_zh: str


@dataclass(frozen=True)
class Module:
    id: str
    title_en: str
    title_zh: str
    status: str
    summary_en: str
    summary_zh: str
    sections: tuple[LabSection, ...] = ()


@dataclass(frozen=True)
class Paper:
    id: str
    code: str
    title_en: str
    title_zh: str
    status: str
    summary_en: str
    summary_zh: str
    modules: tuple[Module, ...] = ()


SIGNAL_ANALYSIS_SECTIONS: tuple[LabSection, ...] = (
    LabSection("home", "Home — Course Overview", "首页 — 课程总览"),
    LabSection("l1", "Lecture 1 — Introduction and Analogue Filters", "第 1 讲 — 引言与模拟滤波器"),
    LabSection("l2", "Lecture 2 — Digital Filters", "第 2 讲 — 数字滤波器"),
    LabSection("l3", "Lecture 3 — Design of Digital Filters", "第 3 讲 — 数字滤波器设计"),
    LabSection("l4", "Lecture 4 — Spectral Estimation", "第 4 讲 — 谱估计"),
    LabSection("practice", "Practice — Concept Checks", "练习 — 概念检查"),
)


IMAGE_PROCESSING_SECTIONS: tuple[LabSection, ...] = (
    LabSection("home", "Home — Course Overview", "首页 — 课程总览"),
    LabSection("l1", "Lecture 1 — Image Enhancement", "第 1 讲 — 图像增强"),
    LabSection("l2", "Lecture 2 — Fourier Analysis", "第 2 讲 — 傅里叶分析"),
    LabSection("l3", "Lecture 3 — Image Restoration", "第 3 讲 — 图像复原"),
    LabSection("l4", "Lecture 4 — Compression and Nonlinear Filters", "第 4 讲 — 压缩与非线性滤波"),
    LabSection("practice", "Practice — Concept Checks", "练习 — 概念检查"),
)


PAPERS: tuple[Paper, ...] = (
    Paper(
        id="b14",
        code="B14",
        title_en="Information Engineering",
        title_zh="信息工程",
        status="available",
        summary_en="Image Processing and Signal Analysis are both available.",
        summary_zh="图像处理和信号分析都可以使用。",
        modules=(
            Module(
                id="image-processing",
                title_en="Image Processing",
                title_zh="图像处理",
                status="available",
                summary_en="B14 Image Processing Lab. Four lectures: enhancement, Fourier analysis, restoration, compression and nonlinear filters.",
                summary_zh="B14 图像处理实验。四讲：增强、傅里叶分析、复原、压缩与非线性滤波。",
                sections=IMAGE_PROCESSING_SECTIONS,
            ),
            Module(
                id="signal-processing",
                title_en="Signal Analysis",
                title_zh="信号分析",
                status="available",
                summary_en="B14 Signal Analysis. Four lectures: analogue filters, digital filters, FIR and IIR design, spectral estimation.",
                summary_zh="B14 信号分析。四讲：模拟滤波器、数字滤波器、FIR 与 IIR 设计、谱估计。",
                sections=SIGNAL_ANALYSIS_SECTIONS,
            ),
        ),
    ),
    Paper(
        id="b15",
        code="B15",
        title_en="Reserved",
        title_zh="预留",
        status="planned",
        summary_en="Placeholder. The paper title and modules will be added with that course.",
        summary_zh="占位。课程名称和模块将在加入该课时再写。",
        modules=(),
    ),
    Paper(
        id="b16",
        code="B16",
        title_en="Reserved",
        title_zh="预留",
        status="planned",
        summary_en="Placeholder. The paper title and modules will be added with that course.",
        summary_zh="占位。课程名称和模块将在加入该课时再写。",
        modules=(),
    ),
)


def paper_by_id(paper_id: str) -> Paper:
    for paper in PAPERS:
        if paper.id == paper_id:
            return paper
    raise KeyError(paper_id)


def module_by_id(paper_id: str, module_id: str) -> Module:
    paper = paper_by_id(paper_id)
    for module in paper.modules:
        if module.id == module_id:
            return module
    raise KeyError(module_id)
