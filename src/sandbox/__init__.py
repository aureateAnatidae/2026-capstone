import numpy as np
from matplotlib import pyplot as plt

from sandbox.config import PLOT
from sandbox.load_data import LoadHassall2026
from sandbox.preprocess import contralateral_rereferencing, repair_artifacts

downsample_sfreq = 500
epoch_length_s = 2


# TODO: Remove sections without music
def main():
    raws = LoadHassall2026()
    X_all, y_all, groups_all = [], [], []

    for i, raw in enumerate(raws):
        raw.resample(sfreq=downsample_sfreq)

        reref_raw = contralateral_rereferencing(raw)

        repair_reref_raw = repair_artifacts(reref_raw)

        filt_repair_reref_raw = repair_reref_raw.copy().filter(
            l_freq=0.1, h_freq=30, picks="eeg"
        )

        X_raw = filt_repair_reref_raw.copy().pick("eeg").apply_baseline((None, None))
        y_raw = filt_repair_reref_raw.copy().pick("Audio")

        if PLOT:
            pass  # TODO: Plot before and after

        X = X_raw.get_data()
        y = y_raw.get_data()

        X_all.append(X)
        y_all.append(y)
        groups_all.append(np.full(len(X), i))
