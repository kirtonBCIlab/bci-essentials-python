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
test_mi.classifier = mi_classifier() # you can add a subset here

# Define the classifier settings
test_mi.classifier.set_mi_classifier_settings(n_splits=5, type="TS", random_seed=35)

# Define channel selection settings
# test_mi.classifier.setup_channel_selection(initial_channels=[], method="SBS", metric="accuracy", max_time=60, n_jobs=-1)
initial_subset=[]
test_mi.classifier.setup_channel_selection(method = "SBFS", metric="accuracy", initial_channels = [],    # wrapper setup
                                max_time= 999, min_channels=1, max_channels=16, performance_delta=-1,      # stopping criterion
                                n_jobs=-1, print_output="verbose", record_performance=True) 

# Load the xdf

test_mi.load_offline_eeg_data(filename  = "examples/data/mi_example_2.xdf", print_output=False) # you can also add a subset here

# Run main loop, this will do all of the classification for online or offline
test_mi.main(online=False, training=True, pp_low=5, pp_high=30, pp_order=5, print_markers=True, print_training=False, print_fit=False, print_performance=True, print_predict=False)


print("debug")