# DONT USE THIS
# NO OFFLINE SWITCH DATA EXISTS YET

import sys
import os

# # Add parent directory to path to access bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))

# from src.bci_data import *
from bci_essentials.bci_data import *
from bci_essentials.visuals import *

# f = open("test.txt", 'w')
# sys.stdout = f

# Initialize data object
test_switch = EEG_data()

# Select a classifier
test_switch.classifier = switch_classifier()
test_switch.classifier.set_switch_classifier_settings(n_splits=2, rebuild=True, random_seed=35)

# Select a file to run, use a file that you have locally
test_switch.load_offline_eeg_data(filename  = "examples/data/July5/AG/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-MISwitch_run-001_eeg.xdf") 

# Run it, brrr brrr
test_switch.main(online=False, training=True)

'''
Results for MISwitch run 1:
1. Neural Network = 90-100%
2. Pyriemannan = 50%
3. Logistic Regression = 70%
'''

