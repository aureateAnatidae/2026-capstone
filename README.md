# The EEG capstone

# Project architecture

The root of this repository is a Python project.

"Sandbox" is a module in which code can be tested before it is compiled into a Jupyter notebook.

### Data

To download data, follow [these instructions](https://openneuro.org/datasets/ds003774/versions/1.0.2/download) or just manually download. The annexed files are required as well. Put the `ds003774-download` directory in the `data/` directory of this repository.

### Google Colab

`scripts/zip_data.py` zips `data/` (following git-annex symlinks, skipping `.git`/`.datalad`) into `data_colab.zip`, rooted at `data/`. Upload it to Google Drive, then in Colab:

```python
# Colab already ships torch, numpy, pandas, matplotlib, scikit-learn and scipy; install the rest.
# `%pip` (not `!pip`) installs into the running kernel's environment.
%pip install -q braindecode mne mne-bids skorch torcheeg soundfile captum

from google.colab import drive
drive.mount("/content/drive")
!unzip -q /content/drive/MyDrive/data_colab.zip -d /content
```

Use `--subjects N` for a small test archive and `-d ds003774-download` to pick one dataset.
