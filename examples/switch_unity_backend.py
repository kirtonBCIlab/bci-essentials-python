import os
import sys

from bci_essentials.bci_data import EEG_data
from bci_essentials.classification import switch_classifier

# # Add parent directory to path to access bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

# Define the SWITCH data object
switch_data = EEG_data()

# LETS TRY IT OUT WITH A WHOLE NEW SWITCH CLASSIFIER
switch_data.classifier = switch_classifier()

switch_data.classifier.set_switch_classifier_settings(
    n_splits=3, rebuild=True, random_seed=35
)
# Connect the streams
switch_data.stream_online_eeg_data()

# Run
switch_data.main(online=True, training=True)
