import os
import sys

# # Add parent directory to path to access bci_essentials
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))

# import bci_essntials
from bci_essentials.bci_data import *

# Initialize the data class
test_ssvep = EEG_data()

# Define a classifier
test_ssvep.classifier = ssvep_basic_classifier()
target_freqs = [14, 13, 12, 11, 10, 9, 8]
test_ssvep.classifier.set_ssvep_settings(n_splits=3, sampling_freq=300, target_freqs = target_freqs, subset=[], random_seed=42, clf_type="Random Forest")

# Connect the streams
test_ssvep.stream_online_eeg_data()

# Run
test_ssvep.main(online=True, training=True)