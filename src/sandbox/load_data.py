"""
Load data from the `data/` directory.

Requires that the `data/` directory is populated with the expected files.
"""

from pathlib import Path

from mne.io import Raw
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids

from sandbox.const import datapath


def LoadHassall2026(
    filepath: Path = datapath / "2026_EEG_PassiveMusicListening_Hassall",
) -> list[Raw]:
    """
    Load passive listening EEG data provided by Cameron Hassall, 2026
    """
    subjects = get_entity_vals(filepath, "subject")
    task = get_entity_vals(filepath, "task")[0]
    datatype = "eeg"

    raws = []
    for s in subjects:
        raw = read_raw_bids(
            BIDSPath(
                subject=s, task=task, root=filepath, datatype=datatype, suffix="eeg"
            )
        )
        # Channels were not labelled with channel types
        raw.set_channel_types({channel: "eeg" for channel in raw.ch_names})
        raw.set_channel_types({"Audio": "misc"})
        raw.load_data()
        raws.append(raw)

    return raws


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    raws = LoadHassall2026()
    raw = raws[0]
    raw.describe()
    raw.plot()
    plt.show()
