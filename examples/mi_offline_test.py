"""
Test Motor Imagery (MI) classification offline using data from an existing stream

"""

import os

from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.paradigm.mi_paradigm import MiParadigm
from bci_essentials.bci_controller import BciController
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.mi_classifier import MiClassifier

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "mi_example_2.xdf")
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)
paradigm = MiParadigm(live_update=True, iterative_training=True)
data_tank = DataTank()

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
test_mi = BciController(classifier, eeg_source, marker_source, paradigm, data_tank)

# Run main loop, this will do all of the classification for online or offline
test_mi.setup(
    online=False,
)
test_mi.run()

print("Debug")
