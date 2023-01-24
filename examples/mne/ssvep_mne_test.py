# This is a script to test the functionality of python SSVEP processing
# Written by Brian Irvine on 08/05/2021

import os
import sys


# Add parent directory to path to access bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),os.pardir))


# # from src.bci_data import *
from bci_essentials.bci_data import *

# import
import matplotlib.pyplot as plt

# Initialize the SSVEP
# should try to automate the reading of some of this stuff from the file header
test_ssvep = EEG_data()

# Define the classifier
test_ssvep.classifier = ssvep_riemannian_mdm_classifier(subset=[])

# Load from xdf into erp_data format
test_ssvep.load_offline_eeg_data(filename = "examples\data\ssvep_example.xdf", format='xdf')

test_ssvep.classifier.set_ssvep_settings(n_splits=3, random_seed=42, n_harmonics=3, f_width=0.5)

# initial_subset=['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Cp4', 'C4', 'F4', 'Cp3', 'C3', 'F3', 'Cz', 'Fz']
# test_ssvep.classifier.setup_channel_selection(method = "SBS", metric="accuracy", initial_channels = initial_subset,    # wrapper setup
#                                 max_time= 999, min_channels=2, max_channels=16, performance_delta=0,      # stopping criterion
#                                 n_jobs=-1, print_output="verbose") 

test_ssvep.main(online=False, training=True, max_samples=5120, pp_type="bandpass", pp_low=3, pp_high=50)


print("debug")
mne_raw = test_ssvep.mne_export_resting_state_as_raw()
mne_raw.plot()

print("debug")
mne_raw = test_ssvep.mne_export_as_raw()
mne_raw.plot()

print("debug")
mne_epochs = test_ssvep.mne_export_as_epochs()
mne_epochs.plot(picks='eeg')



print("debug")
