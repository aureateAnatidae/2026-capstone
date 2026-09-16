import numpy as np
from matplotlib import pyplot as plt
from mne import make_fixed_length_epochs

from sandbox.load_data import LoadHassall2026
from sandbox.preprocess import preprocess_eeg

downsample_sfreq = 500
epoch_length_s = 2


def main():
    raws = LoadHassall2026()
    X_all, y_all, groups_all = [], [], []

    for i, raw in enumerate(raws):
        raw.resample(sfreq=downsample_sfreq)

        epochs = make_fixed_length_epochs(raw, duration=epoch_length_s, preload=True)

        X_raw = epochs.copy().pick("eeg")
        y_raw = epochs.copy().pick("Audio")

        preprocess_eeg(X_raw)

        X = X_raw.get_data()
        y = y_raw.get_data()

        X_all.append(X)
        y_all.append(y)
        groups_all.append(np.full(len(X), i))
