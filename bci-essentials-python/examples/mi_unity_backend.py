import os
import sys

# # Add parent directory to path to access bci_essentials
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))

from bci_essentials.bci_data import *

# Define the MI data object
mi_data = EEG_data()

# Select a classifier
mi_data.classifier = mi_classifier() # you can add a subset here

# Set settings
mi_data.classifier.set_mi_classifier_settings(n_splits=3, type="TS", random_seed=35)

# Connect the streams

mi_data.stream_online_eeg_data() # you can also add a subset here


# Run
mi_data.main(online=True, training=True)


# f.close()