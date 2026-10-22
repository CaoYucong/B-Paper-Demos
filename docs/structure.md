# 文件结构

目录跟着地址走。地址是

`/home/试卷/模块/Lecture/` 或 `/home/试卷/模块/Sheet/`

讲义页内部再按顺序排：首页、第 1 讲到第 4 讲、练习。哈希是实验的英文名去掉空格和标点，例如 `#Bayersampling`。

包名用小写加下划线。地址里的 `ImageProcessing` 来自英文标题，对应的包是 `image_processing`。模块 id 仍是 `image-processing` 和 `signal-processing`。

## 仓库根

```
app.py              Streamlit 实验室。只负责启动和按地址分发，不含数值计算
serve.py            对外服务器，端口 8501。目录页由它发布，实验室挂在 /_lab
src/                课程代码
docs/               和 src 同一棵树的说明
tests/              测试。不放进 src
requirements.txt
pytest.ini          pythonpath 是仓库根
.streamlit/         fileWatcherType = none，改完代码要重启 serve.py
```

## src

```
src/
  catalog.py                 试卷和模块清单。菜单只读这里
  __init__.py
  ui/                        全站外壳。这里没有某一讲的计算
    routes.py                把地址解析成试卷、模块、Lecture 或 Sheet
    site.py                  目录页的 HTML
    shell.py                 Streamlit 里的菜单，以及 render_lecture
    navigation.py            返回、侧栏目录、刷新后滚到哈希
    i18n.py                  中文、English、Bilingual
    anchors.py               实验名变成地址哈希
    components.py            双语标题和版式
    common.py                图像行、提示条、Plotly 图
    visualization.py         图像 PNG 和曲线
  b14/
    image_processing/        /home/B14/ImageProcessing/
      lecture.py             讲义这一整页：侧栏、工作区、按顺序拼各讲
      home.py                首页
      experiments.py         各讲实验的顺序，以及练习跳转 PRACTICE_LINKS
      settings.py            图像实验的默认参数
      metrics.py             MSE、PSNR
      synthetic_images.py    内置图像
      paste_image.py         从剪贴板读图
      paste_image_component/
      lecture1/              增强：采样、点运算、直方图、卷积、Sobel、高斯、匹配滤波
        page.py              这一讲的界面
        fundamentals.py  point_operations.py  histogram.py
        spatial_filters.py  gaussian.py  sobel.py  matched_filter.py
      lecture2/              傅里叶：空间频率、分解、频域滤波、卷积定理、混叠
        page.py
        spatial_frequency.py  fourier.py  frequency_filters.py
        convolution_theorem.py  aliasing.py
      lecture3/              复原
        page.py              把下面几块拼成第 3 讲
        operators.py         周期高斯 PSF、算子 A 和伴随
        degradation.py       退化模型
        wiener.py            维纳滤波和逆滤波
        map_estimation.py    MAP 目标、梯度下降、共轭梯度和傅里叶解
        bayes.py             标量贝叶斯
        comparison.py        三种方法随 λ 的比较
        forward_page.py  operator_page.py  map_page.py
        compare_page.py  bayes_page.py
        motion.py  wiener_gain.py  super_resolution.py
        sampling.py          小网格上的采样、模糊和降采样
        multiframe.py        多帧 MAP、Huber 和一维盲去模糊
        multiframe_page.py   上面两块的界面
        panorama.py
        panorama_frames/     全景用的照片，跟读取它们的代码放在一起
      lecture4/              压缩与非线性滤波
        page.py
        compression.py  huffman.py  dct.py  jpeg_demo.py
        median.py  bilateral.py  nonlocal_means.py  inpainting.py
        window_pick.py       在图上点选或拖动。中心、整行、或小块的左上角
        window_pick_component/
      practice/
        page.py              练习界面
        problems.py          题目和评分。这是自编题，不是官方习题纸
    signal_analysis/         /home/B14/SignalAnalysis/
      lecture.py             讲义这一整页，四讲的界面也写在这个文件里
      experiments.py         各讲实验的顺序
      lecture1/analog.py     模拟滤波器
      lecture2/digital.py    数字滤波器
      lecture3/design.py     FIR 与 IIR 设计
      lecture4/spectral.py   DFT 与谱估计
      practice/problems.py   自编的公式检查
```

图像处理把每一讲的界面放在 `lectureN/page.py`，`lecture.py` 只负责按顺序调用。信号分析的界面仍集中在模块的 `lecture.py`，数值已经按讲分开放。

B15 和 B16 只有 `catalog.py` 里的占位，没有 `src/b15` 或 `src/b16`。Sheet 也没有自己的目录，占位页写在 `src/ui/site.py` 和 `src/ui/shell.py`。

## docs

```
docs/
  architecture.md            服务器、地址和运行时行为
  structure.md               本文件
  b14/
    implementation_plan.md   B14 已经做过的阶段记录
    image_processing/
      source_alignment.md    图像讲义里实际读到的标题
      mathematical_conventions.md
      lecture3/map_notes.md  MAP 推导
    signal_analysis/
      source_alignment.md    信号讲义里实际读到的公式
```

## tests

测试留在 `tests/`，文件名是 `test_` 加主题。导入路径用包的位置，例如 `src.b14.image_processing.lecture3.operators`。

| 文件 | 覆盖 |
| --- | --- |
| `test_routes.py` `test_site.py` | 地址和目录页 |
| `test_i18n.py` | 语言 |
| `test_catalog_practice.py` | 菜单和图像练习评分 |
| `test_lecture1.py` 到 `test_lecture4.py` | 图像四讲的数值 |
| `test_operators.py` `test_degradation.py` `test_map.py` `test_filters.py` `test_bayes.py` `test_images.py` | 第 3 讲求解器和图像工具 |
| `test_lecture3_ext.py` | 运动模糊、全景、超分辨率 |
| `test_multiframe.py` | 多帧采样、伴随、秩、MAP 梯度、Huber 和盲去模糊的核约束 |
| `test_clipboard.py` | 粘贴图像 |
| `test_signal.py` | 信号分析 |

# 维护

加文件时先看它出现在哪一个地址上，再放进对应目录。一次改动只进一个模块。改完目录之后，在同一次改动里更新本文件。

## 放哪里

| 要加的东西 | 放在 |
| --- | --- |
| 试卷、模块的名称和状态 | `src/catalog.py` |
| 地址解析、目录页、语言、两种实验室都用的版式 | `src/ui/` |
| 某一讲的计算 | 该模块的 `lectureN/` |
| 某一讲的界面 | 图像处理用 `lectureN/page.py`。信号分析在界面变长时也拆成 `lectureN/page.py`，由模块的 `lecture.py` 调用 |
| 实验在侧栏和页面上的顺序 | 该模块的 `experiments.py`，按讲义顺序追加 |
| 只被一门课使用的图像、参数、剪贴板 | 该模块目录，不放进 `src/ui/` |
| 照片或其他数据 | 和读取它的那个文件同一目录 |
| 讲义实际写了什么 | `docs/` 下同一门课的 `source_alignment.md` |
| 代码里采用的公式 | 该模块自己的约定文件。图像处理是 `mathematical_conventions.md` |
| 测试 | `tests/test_<主题>.py` |

`src/` 根目录只保留 `catalog.py` 和包入口。某一讲的计算、某一讲的页面，都不放回根目录，也不放进 `src/ui/`。

两门课可以互相引用已经存在的函数，例如第 4 讲使用第 1 讲的高斯滤波。新写的计算仍放在使用它的那一讲；另一讲需要时再导入。

## 加一个实验

1. 在该模块的 `experiments.py` 里，按讲义顺序加一条 `(id, 英文名, 中文名)`。侧栏和标题都读这个元组。
2. 计算放进对应的 `lectureN/`。
3. 在这一讲的界面里画出来，用 `experiment_heading`。锚点由英文名生成，不随语言改变。
4. 图像练习如果要跳到这个实验，在 `PRACTICE_LINKS` 里写下锚点。
5. 给计算加一个 `tests/` 里的测试。

## 加一讲

1. 建 `lectureN/`，并写 `__init__.py`。
2. 界面调用按讲义顺序加进模块的 `lecture.py`。
3. 在 `experiments.py` 增加这一讲的元组。
4. 在 `catalog.py` 的该模块 `sections` 里增加同一讲。侧栏目前读的是 `experiments.py`，`sections` 用来核对讲的集合没有缺。
5. 笔记放进 `docs/` 里同一模块。讲义里没有的内容，在页面上标成补充。

## 加一个模块

1. 在 `src/catalog.py` 的那门试卷下加一个 `Module`。英文标题决定地址，例如 `Signal Analysis` 变成 `/home/B14/SignalAnalysis/`。
2. 建 `src/<试卷>/<模块包>/`，里面放 `lecture.py` 和按讲分开的目录。
3. `src/ui/shell.py` 的 `render_lecture` 在这个模块 `status` 为 `available` 时调用它自己的渲染函数。
4. 建 `docs/<试卷>/<模块包>/`，至少有一份来源对照。
5. 沿用地址、语言和「一页按讲滚动」的方式。界面和计算都写在新包里。

Sheet 仍由外壳渲染占位页。真有习题纸时，再在该模块下建 `sheet/`，并让 `site.py` 与 `shell.py` 在 Sheet 地址上调用它。在那之前不建空目录。

## 加一门试卷

1. 在 `catalog.py` 把占位换成真实名称，并加上模块。B15、B16 在有课程之前保持 Reserved / 预留。
2. 有代码时再建 `src/<试卷 id>/`，有笔记时再建 `docs/<试卷 id>/`。
3. 不要为了对称去复制 `image_processing`。

## 改完要一起动的地方

移动或重命名一个文件时，同时改导入、`tests/` 里的导入，以及本文件里的树。`docs/architecture.md` 只写运行时行为；目录以本文件为准。
