"""
Test P300 offline using data from an existing stream

"""

import os

from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.bci_controller import BciController
from bci_essentials.paradigm.p300_paradigm import P300Paradigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.erp_rg_classifier import ErpRgClassifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# from saving import mne_export_erp_as_epochs

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger(name="p300_mne_plotting")

# Identify the file to simulate
filename = os.path.join("examples\\data", "p300_example.xdf")

# Choose a classifier
classifier = ErpRgClassifier()  # you can add a subset here
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)
paradigm = P300Paradigm()
data_tank = DataTank()

# Set classifier settings
classifier.set_p300_clf_settings(
    n_splits=5,
    lico_expansion_factor=4,
    oversample_ratio=0,
    undersample_ratio=0,
    random_seed=35,
)

# Initialize the ERP data object
test_erp = BciController(classifier, eeg_source, marker_source, paradigm, data_tank)

# Run main loop, this will do all of the classification for online or offline
test_erp.setup()
test_erp.run()

logger.debug("Testing mne_export_erp_as_epochs()")
# mne_epochs = mne_export_erp_as_epochs(test_erp)

# mne_epochs.plot(picks="eeg")

logger.debug("Ran in DEBUG mode")
