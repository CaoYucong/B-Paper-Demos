"""Home page for the Image Processing laboratory."""

from __future__ import annotations

import streamlit as st

from src.ui.i18n import say, t


def render_course_home() -> None:
    st.markdown("<div id='home'></div>", unsafe_allow_html=True)
    st.title(t("B14 Image Processing Lab", "B14 图像处理实验"))
    st.caption(t("Image Analysis · Michaelmas", "图像分析 · Michaelmas"))
    say(
        "The four lectures are one pipeline, and they follow on this page in the order of the slides. "
        "Enhancement and filtering say what a local operation does. "
        "Fourier analysis says the same operation as a product of spectra. Restoration asks how to undo a "
        "filter once noise has been added. Compression and nonlinear filters then ask which structure can "
        "be thrown away, and which operations cannot be written as convolution.",
        "四讲是一条链，并按讲义顺序排在这一页上。增强和滤波说明局部运算做什么。傅里叶分析把同一个运算写成频谱相乘。复原要问：加上噪声之后，怎样撤销一个滤波器。压缩和非线性滤波接着问：哪些结构可以丢掉，哪些运算不能写成卷积。",
    )
    columns = st.columns(4)
    cards = [
        (
            "Lecture 1",
            "第 1 讲",
            "Image Enhancement",
            "图像增强",
            "An image is an array. Point operations change one sample. Correlation and convolution mix a neighbourhood.",
            "图像是数组。点运算只改一个样本。相关和卷积混合一个邻域。",
        ),
        (
            "Lecture 2",
            "第 2 讲",
            "Fourier Analysis",
            "傅里叶分析",
            "A sinusoid is one frequency. Filtering is multiplication. Downsampling without a prefilter aliases.",
            "正弦是一个频率。滤波是相乘。不做预滤波就降采样会产生混叠。",
        ),
        (
            "Lecture 3",
            "第 3 讲",
            "Image Restoration",
            "图像复原",
            "g = A f + n. Inverse filtering divides by H. Wiener and quadratic MAP keep that division stable.",
            "g = A f + n。逆滤波除以 H。维纳滤波和二次 MAP 让这个除法保持稳定。",
        ),
        (
            "Lecture 4",
            "第 4 讲",
            "Compression and Nonlinear Filters",
            "压缩与非线性滤波",
            "Storage, RLE, Huffman, a JPEG-like DCT pipeline, then median, bilateral, non-local means, and inpainting.",
            "存储、游程编码、霍夫曼、类 JPEG 的 DCT 流程，然后是中值、双边、非局部均值和修补。",
        ),
    ]
    for column, (lecture_en, lecture_zh, title_en, title_zh, body_en, body_zh) in zip(columns, cards):
        with column:
            with st.container(border=True):
                st.caption(t(lecture_en, lecture_zh))
                st.markdown(f"**{t(title_en, title_zh)}**")
                say(body_en, body_zh)
    st.markdown(f"##### {t('How the workspace is shared', '工作区如何共用')}")
    say(
        "Choose a built-in image, an upload, or the workspace in the sidebar. "
        "A result can be saved with **Use as workspace image**. Switch the source to Workspace before opening the next lecture. "
        "Restoration experiments never pass the original image into the solver; it is used only to form the observation and to score error.",
        "在侧栏选择内置图像、上传或工作区。结果可以用「用作工作区图像」保存。打开下一讲之前，把来源换成工作区。复原算法不会把原图送进求解器；原图只用来生成观测和计算误差。",
    )
    st.markdown(f"##### {t('What is additional', '哪些是补充说明')}")
    say(
        "Matched filtering is an additional detection example. The integer-shift super-resolution page is a supplementary illustration: "
        "the lecture titles the full sketch solution as not examinable. The JPEG page reconstructs an array; it does not write a JPEG file.",
        "匹配滤波是补充的检测例子。整数位移超分辨率是补充演示：讲义把完整的草图解标成非考试内容。JPEG 页面重建的是数组，不会写出 JPEG 文件。",
    )
