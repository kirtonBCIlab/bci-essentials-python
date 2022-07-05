# This is a script to test the functionality of python SSVEP processing
# Written by Brian Irvine on 08/05/2021

import os
import sys


# # Add parent directory to path to access bci_essentials
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))



# # from src.bci_data import *
from bci_essentials.bci_data import *

# import
import matplotlib.pyplot as plt

# Initialize the SSVEP
# should try to automate the reading of some of this stuff from the file header
test_ssvep = EEG_data()

# Load from xdf into erp_data format
test_ssvep.load_offline_eeg_data(filename = "examples/data/ssvep_example.xdf", format='xdf')

# Define the classifier

test_ssvep.classifier = ssvep_basic_classifier(subset=[])

# 
target_freqs = [9, 9.6, 10.28, 11.07, 12, 13.09, 14.4]
# TODO add the frequencies from the marker string to target freqs

#test_ssvep.classifier.set_ssvep_settings(sampling_freq = 300, target_freqs = target_freqs)

test_ssvep.classifier.set_ssvep_settings(n_splits=3, sampling_freq=300, target_freqs = target_freqs, random_seed=42, clf_type="Random Forest")

#test_ssvep.classifier.set_ssvep_settings(n_splits=3, sampling_freq=300, target_freqs = target_freqs, subset=[], random_seed=42, clf_type="LDA")

test_ssvep.main(online=False, training=True, max_samples= 5120)


# # Plot the EEG for inspection
# eeg = np.array(test_ssvep.eeg_data)
# t = test_ssvep.eeg_timestamps

# print(eeg.shape)
# nsamples, nchannels = eeg.shape

# for i in range(nchannels):
#     plt.plot(t, eeg[:,i])

# plt.show()
