# B Paper Demonstrations

B Paper Demonstrations is a local laboratory for Oxford Engineering B-paper material.

B14 has two laboratories. Image Processing is `/home/B14/ImageProcessing/Lecture/`. Signal Analysis, from the Michaelmas signal lectures, is `/home/B14/SignalAnalysis/Lecture/`. Each topic has a Sheet page; those sheets are placeholders. B15 and B16 are reserved. 

## Addresses

`python serve.py` listens on port 8501

## Install and launch

Python 3.11 or newer. From this directory:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pytest
python serve.py
```

Dependencies are NumPy, SciPy, Matplotlib, Pillow, Plotly, Streamlit, and pytest. No GPU and no PyTorch. The program does not call an external model service.

The interface language is 中文, English, or Bilingual. The default is Bilingual.

Lecture 3 keeps the original degradation, Wiener, and MAP demonstrations. After the panorama photographs, a synthetic grid shows why shifted low-resolution frames add samples, what the matrix M does, and how multi-frame MAP searches for a high-resolution image. Blind deblurring is a short one-dimensional demo and is marked not examinable.

The file tree and the rules for extending it are in `docs/structure.md`.

