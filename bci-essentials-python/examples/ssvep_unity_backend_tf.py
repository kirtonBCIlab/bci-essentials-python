import os
import sys

# # Add parent directory to path to access bci_essentials
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))

from bci_essentials.bci_data import *

# Initialize the EEG Data
test_ssvep = EEG_data()

# set train complete to true so that predictions will be allowed
test_ssvep.train_complete = True

# Define the classifier
test_ssvep.classifier = ssvep_basic_classifier_tf()
target_freqs = [9, 9.6, 10.28, 11.07, 12, 13.09, 14.4] 

# Connect the streams
test_ssvep.stream_online_eeg_data()

# Run
test_ssvep.main(online=True, training=False, train_complete=True)