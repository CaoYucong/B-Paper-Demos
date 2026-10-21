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

