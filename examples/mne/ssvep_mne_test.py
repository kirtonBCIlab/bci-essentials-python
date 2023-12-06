# This is a script to test the functionality of python SSVEP processing
# Written by Brian Irvine on 08/05/2021

import os

from bci_essentials.sources.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.eeg_data import EEG_data
from bci_essentials.classification.ssvep_riemannian_mdm_classifier import (
    SSVEP_riemannian_mdm_classifier,
)

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "ssvep_example.xdf")
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)

# Define the classifier
classifier = SSVEP_riemannian_mdm_classifier(subset=[])
classifier.set_ssvep_settings(n_splits=3, random_seed=42, n_harmonics=3, f_width=0.5)

# Initialize the SSVEP
# should try to automate the reading of some of this stuff from the file header
test_ssvep = EEG_data(classifier, eeg_source, marker_source)

# initial_subset=['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Cp4', 'C4', 'F4', 'Cp3', 'C3', 'F3', 'Cz', 'Fz']
# test_ssvep.classifier.setup_channel_selection(method = "SBS", metric="accuracy", initial_channels = initial_subset,    # wrapper setup
#                                 max_time= 999, min_channels=2, max_channels=16, performance_delta=0,      # stopping criterion
#                                 n_jobs=-1, print_output="verbose")

test_ssvep.main(
    online=False,
    training=True,
    max_samples=5120,
    pp_type="bandpass",
    pp_low=3,
    pp_high=50,
)


print("debug")
mne_raw = test_ssvep.mne_export_resting_state_as_raw()
mne_raw.plot()

print("debug")
mne_raw = test_ssvep.mne_export_as_raw()
mne_raw.plot()

print("debug")
mne_epochs = test_ssvep.mne_export_as_epochs()
mne_epochs.plot(picks="eeg")


print("debug")
