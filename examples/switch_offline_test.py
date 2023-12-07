import os

from bci_essentials.sources.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.eeg_data import EEG_data
from bci_essentials.classification.switch_mdm_classifier import Switch_mdm_classifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger()
logger.debug("Running %s", __file__)

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "switch_example.xdf")
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)

# Select a classifier
classifier = Switch_mdm_classifier()
classifier.set_switch_classifier_settings(n_splits=2, rebuild=True, random_seed=35)

# Initialize data object
test_switch = EEG_data(classifier, eeg_source, marker_source)

# Run it, brrr brrr
test_switch.main(online=False, training=True)

"""
Results for switch example:
1. Neural Network = 80-100%
2. Pyriemannan = 50-75%
3. Logistic Regression = 40-60%
"""
