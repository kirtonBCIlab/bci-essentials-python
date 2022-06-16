import sys

# from src.bci_data import *
from bci_essentials.bci_data import *
from bci_essentials.visuals import *

# f = open("test.txt", 'w')
# sys.stdout = f

# Initialize data object
test_switch = EEG_data()

# Select a classifier
test_switch.classifier = switch_classifier()
test_switch.classifier.set_switch_classifier_settings(n_splits=3, rebuild=True, random_seed=35)

# Select a file to run, use a file that you have locally
test_switch.load_offline_eeg_data(filename  = "C:/Users/brian/Documents/OptimizationStudy/TeamData/May5/BI/MI/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-GamifiedMI_run-002_eeg.xdf")

# Run it, brrr brrr
test_switch.main(online=False, training=True)