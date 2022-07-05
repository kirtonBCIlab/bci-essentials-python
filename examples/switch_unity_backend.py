import os
import sys

# # Add parent directory to path to access bci_essentials
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))

from bci_essentials.bci_data import *
# f = open("test.txt", 'w')
# sys.stdout = f

# Define the sWITCH data object
switch_data = EEG_data()


# # LETS TRY IT OUT WITH JUST THE MI CLASSIFIER
# # Select a classifier
# switch_data.classifier = mi_classifier()

# switch_data.classifier.set_mi_classifier_settings(n_splits=3, type="TS", subtract_center=False, rebuild=True, random_seed=35)
# # Connect the streams
# switch_data.stream_online_eeg_data()

# OR


# LETS TRY IT OUT WITH A WHOLE NEW SWITCH CLASSIFIER
switch_data.classifier = switch_classifier()

switch_data.classifier.set_switch_classifier_settings(n_splits=3, rebuild=True, random_seed=35)
# Connect the streams
switch_data.stream_online_eeg_data()


# Run
switch_data.main(online=True, training=True)


# f.close()
