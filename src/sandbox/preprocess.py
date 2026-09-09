"""
Perform preprocessing similar to prior capstone projects.

1. Resample data
2. Perform contralateral referencing
- https://mne.tools/stable/auto_examples/preprocessing/contralateral_referencing.html
3. Band-pass 0.2hz, 60hz
"""

import mne
from mne.io import Raw


def preprocess(raw: Raw):

    # this splits electrodes into 3 groups; left, midline, and right
    ch_names = mne.channels.make_1020_channel_selections(raw.info, return_ch_names=True)

    # remove the ref channels from the lists of to-be-rereferenced channels
    ch_names["Left"].remove("M1")
    ch_names["Right"].remove("M2")
    # midline referencing to mean of mastoids:
    mastoids = ["M1", "M2"]
    rereferenced_midline_chs = (
        raw.copy()
        .pick(mastoids + ch_names["Midline"])
        .set_eeg_reference(mastoids)
        .drop_channels(mastoids)
    )

    # contralateral referencing (alters channels in `raw` in-place):
    for ref, hemi in dict(M2=ch_names["Left"], M1=ch_names["Right"]).items():
        mne.set_bipolar_reference(
            raw, anode=hemi, cathode=[ref] * len(hemi), copy=False
        )
    # strip off '-M1' and '-M2' suffixes added to each bipolar-referenced channel
    raw.rename_channels(lambda ch_name: ch_name.split("-")[0])

    # replace unreferenced midline with rereferenced midline
    _ = raw.drop_channels(ch_names["Midline"]).add_channels([rereferenced_midline_chs])
    raw.set_montage("standard_1020", on_missing="warn")

    
