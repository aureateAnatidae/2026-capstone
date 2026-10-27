import numpy as np
from matplotlib import pyplot as plt

from sandbox.config import PLOT
from sandbox.load_data import LoadHassall2026
from sandbox.load_data import LoadMUSIN_G
from sandbox.preprocess import contralateral_rereferencing, repair_artifacts

downsample_sfreq = 250
# epoch_length_s = 2


# TODO: Remove sections without music
def main():
    df = LoadMUSIN_G()
    X_all, y_all, groups_all = [], [], []

