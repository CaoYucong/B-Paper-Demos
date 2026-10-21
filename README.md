# B Paper Demonstrations

B Paper Demonstrations is a local laboratory for Oxford Engineering B-paper material. The public pages are directories. 

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

