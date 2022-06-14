"""
Test Motor Imagery (MI) classification offline using data from an existing stream

"""

import os
import sys

# Add parent directory to path to access bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))

# from src.bci_data import *
from bci_essentials.bci_data import *
from bci_essentials.visuals import *

# Initialize data object
test_mi = EEG_data()

# Select a classifier
test_mi.classifier = mi_classifier()

# Define the classifier settings
test_mi.classifier.set_mi_classifier_settings(n_splits=3, type="TS", subtract_center=False, rebuild=True, random_seed=35)

# Load the xdf
test_mi.load_offline_eeg_data(filename  = "examples/data/mi_example.xdf")

# Run main loop, this will do all of the classification for online or offline
test_mi.main(online=False, training=True)