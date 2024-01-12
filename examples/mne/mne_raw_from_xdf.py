import os
import numpy as np
import mne

from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the script
logger = Logger(name="mne_raw_from_xdf")

# Identify the file to simulate
filename = os.path.join("examples\\data", "mi_example.xdf")

# Load the example EEG / marker streams
marker_source = XdfMarkerSource(filename)
eeg_source = XdfEegSource(filename)

# Get the data from that stream
marker_data, marker_timestamps = marker_source.get_markers()
eeg_data, eeg_timestamps = eeg_source.get_samples()

# Parse EEG to MNE
info = mne.create_info(
    eeg_source.channel_labels, eeg_source.fsample, ["eeg"] * eeg_source.n_channels
)
raw = mne.io.RawArray(np.transpose(eeg_data[:, :16]), info)
raw.filter(l_freq=0.1, h_freq=15)
mne.set_eeg_reference(raw, ref_channels="average")

logger.info("MNE information: %s", info)
