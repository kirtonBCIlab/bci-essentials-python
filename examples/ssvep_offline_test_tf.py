# This is a script to test the functionality of python SSVEP processing
# Written by Brian Irvine on 08/05/2021

import os
import sys

from ..bci_essentials.bci_data import EEG_data
from ..bci_essentials.classification.ssvep_basic_tf_classifier \
    import SSVEP_basic_tf_classifier

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "ssvep_example.xdf")

# Initialize the SSVEP
# should try to automate the reading of some of this stuff from the file header
test_ssvep = EEG_data()

# Define the classifier
test_ssvep.classifier = SSVEP_basic_tf_classifier(subset=[])

# Load from xdf into erp_data format
test_ssvep.load_offline_eeg_data(
    filename=filename, format="xdf"
)

test_ssvep.classifier.set_ssvep_settings(
    sampling_freq=256,
    target_freqs=[
        "24",
        "20.57143",
        "18",
        "16",
        "14.4",
        "12",
        "10.28571",
        "9",
        "8",
        "7.2",
        "6",
        "4.965517",
    ],
)

test_ssvep.main(
    online=False,
    training=False,
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
