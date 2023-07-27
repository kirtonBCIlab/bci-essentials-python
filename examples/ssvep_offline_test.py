# This is a script to test the functionality of python SSVEP processing
# Written by Brian Irvine on 08/05/2021

import os
import sys


# Add parent directory to path to access bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


# # from src.bci_data import *
from bci_essentials.bci_data import *
from bci_essentials.classification import SSVEP_riemannian_mdm_classifier

# import
import matplotlib.pyplot as plt

# Initialize the SSVEP
# should try to automate the reading of some of this stuff from the file header
test_ssvep = EEG_data()

# Define the classifier
test_ssvep.classifier = SSVEP_riemannian_mdm_classifier(subset=[])

# Load from xdf into erp_data format
test_ssvep.load_offline_eeg_data(
    filename="examples\data\ssvep_example.xdf", format="xdf"
)

test_ssvep.classifier.set_ssvep_settings(
    n_splits=3, random_seed=42, n_harmonics=3, f_width=0.5
)

# initial_subset=['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Cp4', 'C4', 'F4', 'Cp3', 'C3', 'F3', 'Cz', 'Fz']
# test_ssvep.classifier.setup_channel_selection(method = "SBS", metric="accuracy", initial_channels = initial_subset,    # wrapper setup
#                                 max_time= 999, min_channels=2, max_channels=16, performance_delta=0,      # stopping criterion
#                                 n_jobs=-1, print_output="verbose")

test_ssvep.main(
    online=False,
    training=True,
    max_samples=5120,
    pp_type="bandpass",
    pp_low=3,
    pp_high=50,
)

print("debug")

# Some optional plotting
# # plot a spectrogram of the session
# for ci, ch in enumerate(test_ssvep.channel_labels):
#     eeg = np.array(test_ssvep.classifier.X[0,ci,:])
#     tv = [e/test_ssvep.fsample for e in list(range(0,len(eeg)))]

#     plt.plot(tv, eeg)

# plt.show()
# plt.clf()

# for i in range(48):
#     eeg = np.array(test_ssvep.classifier.X[i,15,:])
#     tv = [e/test_ssvep.fsample for e in list(range(0,len(eeg)))]

#     f, t, Sxx = scipy.signal.spectrogram(eeg, fs=test_ssvep.fsample, nperseg=512)

#     fp, Pxx = scipy.signal.welch(eeg, fs=test_ssvep.fsample, nperseg=512, return_onesided = True)

#     f_target = test_ssvep.classifier.target_freqs[int(test_ssvep.classifier.y[i])]

#     # Plot the EEG for inspection
#     plt.subplot(311)
#     plt.plot(tv,eeg)
#     plt.title(f_target)

#     # psd
#     plt.subplot(312)
#     plt.plot(fp, Pxx)
#     plt.vlines(x=[f_target,f_target*2,f_target*3], ymin=-1000, ymax=1000)
#     plt.xlim([-0.5,50])
#     plt.ylim([-0.5,20])

#     # spectrogram
#     plt.subplot(313)
#     plt.pcolormesh(t, f, Sxx, shading='gouraud')
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')
#     plt.ylim([0,30])
#     plt.xlim([0,4])
#     plt.hlines(y=[f_target,f_target*2,f_target*3], xmin=-5, xmax=5, color='r')
#     plt.show()

#     print("debug")

#     # clear
#     plt.clf()
