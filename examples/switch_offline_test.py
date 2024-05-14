import os

from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.eeg_data import EegData
from bci_essentials.paradigm.mi_paradigm import MiParadigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.switch_mdm_classifier import SwitchMdmClassifier

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "switch_example.xdf")
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)
paradigm = MiParadigm(live_update=True, iterative_training=True)
data_tank = DataTank()

# Select a classifier
classifier = SwitchMdmClassifier()
classifier.set_switch_classifier_mdm_settings(n_splits=2, rebuild=True, random_seed=35)

# Initialize data object
test_switch = EegData(classifier, eeg_source, marker_source, paradigm, data_tank)

# Run it, brrr brrr
test_switch.setup(online=False)
test_switch.run()

"""
Results for switch example:
1. Neural Network = 80-100%
2. Pyriemannan = 50-75%
3. Logistic Regression = 40-60%
"""
