"""
Test Motor Imagery (MI) classification offline using data from an existing stream

"""

import os
import sys

# Add parent directory to path to access bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from bci_essentials.bci_data import EEG_data
from bci_essentials.classification import MI_classifier

# Initialize data object
test_mi = EEG_data()

# Select a classifier
test_mi.classifier = MI_classifier()  # you can add a subset here

# Define the classifier settings
test_mi.classifier.set_mi_classifier_settings(
    n_splits=5,
    type="TS",
    random_seed=35,
    channel_selection="riemann",
    covariance_estimator="oas",
)

# Load the xdf
test_mi.load_offline_eeg_data(
    filename="examples/data/mi_example_2.xdf", print_output=False
)  # you can also add a subset here

# Run main loop, this will do all of the classification for online or offline
test_mi.main(
    online=False,
    training=True,
    pp_low=5,
    pp_high=50,
    pp_order=5,
    print_markers=False,
    print_training=False,
    print_fit=False,
    print_performance=True,
    print_predict=False,
)


print("debug")
