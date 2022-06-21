import os
import sys

from pylsl import StreamInlet, resolve_stream, resolve_byprop

# Add parent directory to path to access bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))

from bci_essentials.bci_data import *


# f = open("test.txt", 'w')
# sys.stdout = f

# Define the MI data object
mi_data = EEG_data()

# Select a classifier
mi_data.classifier = mi_classifier()
#mi_data.classifier.set_mi_classifier_settings(n_splits=5, type="TS", subset=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], pred_threshold = 0.5, subtract_center=False, rebuild=True, random_seed=35)
mi_data.classifier.set_mi_classifier_settings(n_splits=3, type="TS", subtract_center=False, rebuild=True, random_seed=35)
# Connect the streams
#mi_data.stream_online_eeg_data(subset= ['AF3', 'F7', 'F3', 'C1', 'C3', 'FC5', 'T7', 'CP1', 'P7', 'O1', 'O2', 'P8', 'CP2', 'T8', 'FC6', 'C4', 'C2', 'F4', 'F8'])
mi_data.stream_online_eeg_data(subset= ['AF3', 'F7', 'F3', 'C1', 'C3', 'FC5', 'T7', 'CP1', 'P7', 'O1', 'O2', 'P8', 'CP2', 'T8', 'FC6', 'C4', 'C2', 'F4', 'F8', 'AF4'])
# Run
mi_data.main(online=True, training=True)


# f.close()