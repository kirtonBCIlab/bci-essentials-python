"""
Test Motor Imagery (MI) classification offline using data from an existing stream

"""

import os

from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.eeg_data import EegData
from bci_essentials.classification.mi_classifier import MiClassifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger()

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "mi_example_2.xdf")
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)

# Select a classifier
classifier = MiClassifier()  # you can add a subset here

# Define the classifier settings
classifier.set_mi_classifier_settings(
    n_splits=5,
    type="TS",
    random_seed=35,
    channel_selection="riemann",
    covariance_estimator="oas",
)

# Initialize data object
test_mi = EegData(classifier, eeg_source, marker_source)

# Run main loop, this will do all of the classification for online or offline
test_mi.setup(
    online=False,
    training=True,
    pp_low=5,
    pp_high=50,
    pp_order=5,
)
test_mi.run()

logger.debug("Ran in debug mode")
