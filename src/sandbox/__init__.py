from matplotlib import pyplot as plt

from sandbox.load_data import LoadHassall2026
from sandbox.preprocess import preprocess


def main():
    raws = LoadHassall2026()
    x, y = raws[0].get_data(picks=["eeg"]), raws[0].get_data(picks=['Audio'])
    plt.plot(y[0])
    plt.show()
    # print(y[0,:100])
    print(type(x))

    # preprocess(raws[0])
    # raws[0].plot_sensors(show_names=True, sphere="eeglab")
    # plt.show()
