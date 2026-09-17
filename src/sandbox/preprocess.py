import mne
from matplotlib import pyplot as plt
from mne.io import Raw
from mne.preprocessing import ICA, create_eog_epochs

from sandbox.config import PLOT, RANDOM_STATE


def contralateral_rereferencing(raw: Raw) -> Raw:
    """
    Rereference all channels of a mne.io.Raw object to M1 and M2
    Note: in data from Hassall 2026, channels TP9 and TP10 are already renamed to M1 and M2

    https://mne.tools/stable/auto_examples/preprocessing/contralateral_referencing.html
    """
    reref_raw = raw.copy()

    # this splits electrodes into 3 groups; left, midline, and right
    ch_names = mne.channels.make_1020_channel_selections(
        reref_raw.info, return_ch_names=True
    )

    # remove the ref channels from the lists of to-be-rereferenced channels
    ch_names["Left"].remove("M1")
    ch_names["Right"].remove("M2")
    # midline referencing to mean of mastoids:
    mastoids = ["M1", "M2"]
    rereferenced_midline_chs = (
        reref_raw.copy()
        .pick(mastoids + ch_names["Midline"])
        .set_eeg_reference(mastoids)
        .drop_channels(mastoids)
    )

    # contralateral referencing (alters channels in `raw` in-place):
    for ref, hemi in {"M2": ch_names["Left"], "M1": ch_names["Right"]}.items():
        mne.set_bipolar_reference(
            reref_raw, anode=hemi, cathode=[ref] * len(hemi), copy=False
        )
    # strip off '-M1' and '-M2' suffixes added to each bipolar-referenced channel
    reref_raw.rename_channels(lambda ch_name: ch_name.split("-")[0])

    # replace unreferenced midline with rereferenced midline
    _ = reref_raw.drop_channels(ch_names["Midline"]).add_channels(
        [rereferenced_midline_chs]
    )
    reref_raw.set_montage("standard_1020", on_missing="warn")

    return reref_raw


def repair_artifacts(raw: Raw) -> Raw:
    """
    Perform ICA, remove EOG/ECG components from the signal.

    https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
    """
    eog_evoked = create_eog_epochs(raw, ch_name=["Fp1", "Fp2"]).average()
    eog_evoked.apply_baseline(baseline=(None, -0.2))
    if PLOT:
        eog_evoked.plot_joint()

    # A copy should be made. Filtering below 1.0 hz makes it harder for ICA to find components.
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)

    ica = ICA(n_components=10, max_iter=35000, random_state=RANDOM_STATE)
    ica.fit(filt_raw)

    if PLOT:
        ica.plot_components()

    explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
    for channel_type, ratio in explained_var_ratio.items():
        print(
            f"Fraction of {channel_type} variance explained by all components: {ratio}"
        )

    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(
        filt_raw, ch_name=["Fp1", "Fp2"], threshold=2.0
    )
    ica.exclude = eog_indices

    if PLOT & bool(eog_indices):
        # barplot of ICA component "EOG match" scores
        ica.plot_scores(eog_scores)

        # plot diagnostics
        ica.plot_properties(filt_raw, picks=eog_indices)

        # plot ICs applied to raw data, with EOG matches highlighted
        ica.plot_sources(filt_raw, show_scrollbars=False)

        # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
        ica.plot_sources(eog_evoked)
    else:
        print("No EOG-related components were found.")

    ica_raw = raw.copy()
    ica.apply(ica_raw)

    return ica_raw
