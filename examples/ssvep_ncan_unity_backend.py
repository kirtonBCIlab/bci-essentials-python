# import bci_essntials
import numpy as np
from bci_essentials.eeg_data import EEG_data
from bci_essentials.classification.SSVEP_NCAN_classifier import SSVEP_NCAN_classifier


# Define the classifier
initial_subset = ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Cp4', 'C4', 'F4', 'Cp3', 'C3', 'F3', 'Cz', 'Fz']
classifier = SSVEP_NCAN_classifier(
    subset = initial_subset
    )
classifier.set_ssvep_settings(
    sampling_freq = 128,
    target_freqs = np.array([
        6.111111,
        9.705882,
        12.69231
        ]),
    classifier_name = "CCA",
    harmonics_count = 3
)

# Initialize the data class
test_ssvep = EEG_data()

# # Connect the streams
test_ssvep.stream_online_eeg_data()

# # Channel Selection

# initial_subset=[]
# test_ssvep.classifier.setup_channel_selection(method = "SBFS", metric="accuracy", initial_channels = initial_subset,    # wrapper setup
#                                 max_time= 999, min_channels=2, max_channels=14, performance_delta=0,      # stopping criterion
#                                 n_jobs=-1, print_output="verbose")

test_ssvep.main(
    online=True,
    training=True,
    max_samples=5120,
    pp_type="bandpass",
    pp_low=3,
    pp_high=50,
)

print("debug")
