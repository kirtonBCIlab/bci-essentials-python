import os
import sys

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
mi_data.classifier.set_mi_classifier_settings(n_splits=3, type="TS", random_seed=35)
# Connect the streams
mi_data.stream_online_eeg_data()

# Run
mi_data.main(online=True, training=True)


# f.close()