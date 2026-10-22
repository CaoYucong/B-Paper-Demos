# Architecture

`serve.py` is the public server. `app.py` is the Streamlit laboratory and
does not contain numerical code.

## Navigation

`src/catalog.py` is the list of papers and modules. `src/ui/routes.py`
turns that list into a directory address:

`/home/B14/ImageProcessing/Lecture/#Bayersampling`

`serve.py` publishes those directories. A refresh keeps the directory and
the hash. The hash is the experiment name with spaces and punctuation
removed. The Image Processing lecture page frames the Streamlit laboratory
at `/_lab`, because Streamlit cannot own a nested pathname.

- B14 is available. Image Processing and Signal Analysis both open a
  laboratory. Signal Analysis is the Michaelmas signal course; its directory
  is `/home/B14/SignalAnalysis/Lecture/`.
- B15 and B16 are reserved papers with no invented course titles.

Each module has two tracks. `Lecture` opens the laboratory when that
module is available, and a reserved page otherwise. `Sheet` is a
placeholder on every module.

Image Processing and Signal Analysis each scroll Home, Lectures 1–4, and
Practice. The sidebar links jump to those sections on the Lecture page.

`docs/structure.md` is the file tree and the rules for adding a paper, a
module, a lecture, or an experiment. Adding a module still means an entry
in `src/catalog.py`, a package under that paper, and a branch in
`render_lecture`. The new laboratory keeps its own pages.

## Shared image

The sidebar chooses a built-in image, an upload, or the workspace.
`Use as workspace image` stores a `float64` array. Another lecture sees it
only after the source is set to Workspace. Resizing a stored array uses a
linear zoom and does not change the stored copy.

Restoration solvers still receive only the observation, the operator, and
the regularisation parameter.

## Tests

`pytest` collects the original MAP tests and `tests/test_lecture1.py`
through `tests/test_lecture4.py`, plus the catalog and practice grader.
A module is not described as tested unless that file has been run.
