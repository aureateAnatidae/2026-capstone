"""
Load data from the `data/` directory.

Requires that the `data/` directory is populated with the expected files.
"""

from math import gcd
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.io import Raw
from mne_bids import BIDSPath, find_matching_paths, get_entity_vals, read_raw_bids
from scipy.io import wavfile
from scipy.signal import resample_poly

from sandbox.config import DATA_PATH


def LoadHassall2026(
    filepath: Path = DATA_PATH / "2026_EEG_PassiveMusicListening_Hassall",
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


def add_wav_as_misc(raw, wav_path, ch_name="Audio", mono="mean", offset=10.0):
    """Append a wav file's audio to an mne.io.Raw as a misc channel.

    The audio starts `offset` seconds into the Raw; the preceding samples are silence.
    """
    sfreq_wav, data = wavfile.read(wav_path)

    # Convert integer PCM to float in [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)

    # Collapse stereo to mono
    if data.ndim == 2:
        data = data.mean(axis=1) if mono == "mean" else data[:, 0]

    # Resample to the Raw's sampling rate
    sfreq_raw = raw.info["sfreq"]
    if sfreq_wav != sfreq_raw:
        up, down = int(round(sfreq_raw)), int(sfreq_wav)
        g = gcd(up, down)
        data = resample_poly(data, up // g, down // g)

    # Prepend silence for the offset
    n_silence = int(round(offset * sfreq_raw))
    data = np.concatenate([np.zeros(n_silence), data])

    # Match length to the Raw (trim or zero-pad at the end)
    n = raw.n_times
    if len(data) >= n:
        data = data[:n]
    else:
        data = np.pad(data, (0, n - len(data)))

    info = mne.create_info([ch_name], sfreq_raw, ch_types="misc")
    audio_raw = mne.io.RawArray(data[np.newaxis, :], info, first_samp=raw.first_samp)

    raw.load_data()
    raw.add_channels([audio_raw], force_update_info=True)
    return raw


def match_song(subject, session, task, extension, run, suffix, root):
    event_file = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        extension=extension,
        run=run,
        suffix=suffix,
        root=root,
        datatype="eeg",
    )
    ev = pd.read_csv(event_file, sep="\t")
    on = ev.loc[ev["value"] == "stm+", "onset"].to_numpy()
    off = ev.loc[ev["value"] == "fxnd", "onset"].to_numpy()
    song_len = off - on  # 12 lengths, in presentation order

    print("The length of this song is:", song_len)
    # yield song_len


def LoadMUSIN_G(
    filepath: Path = DATA_PATH / "ds003774-download",
) -> list[Raw]:
    """
    Load data from https://openneuro.org/datasets/ds003774/versions/1.0.2

    Find the event markers from the event .tsv file
    Select EEG from the start of the song to the end of the song

    A subject's .tsv file contains all start and end markers for each song.

    Note:
        - for 11 participants, the sampling rate of EEG is 1000Hz, for the remaining 9, 250Hz.
        - none of the sessions are labelled with the song, but we do have the length of the session and the length of the song
            - yes, we match the song to the session by their length
            - lengths of the songs are included in an unformatted text file, which has been interpreted in the function
                `match_song()`
    """
    subjects = get_entity_vals(filepath, "subject")
    sessions = get_entity_vals(filepath, "session")

    task = get_entity_vals(filepath, "task")[0]
    datatype = "eeg"

    songs = {
        1: "Trip to the lonely planet",
        2: "Sail",
        3: "Concept 15",
        4: "Aurore",
        5: "Proof",
        6: "Glider",
        7: "Raag Bihag",
        8: "Albela sajan",
        9: "Mor Bani thanghat Kare",
        10: "Fallin",
        11: "Master of Running",
        12: "JB",
    }

    raws = []
    for sub in subjects[1:5]:
        runs = get_entity_vals(root=filepath / f"sub-{sub}", entity_key="run")

        songs = match_song(sub, "01", task, ".tsv", "1", "events", filepath)

        for ses in sessions:
            # The 1st session corresponds to the 1st run, which corresponds to the 1st instance of 'stm+' and 'fxnd' in the .tsv
            paths = find_matching_paths(
                subjects=sub,
                sessions=ses,
                tasks=task,
                extensions=[".set"],
                suffixes="eeg",
                root=filepath,
            )
            song_num = int(ses)
            song_file = filepath / "Code" / "ESongs" / f"{song_num}.esh.wav"
            

            for path in paths:
                raw = read_raw_bids(path)
                add_wav_as_misc(raw, song_file)
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
