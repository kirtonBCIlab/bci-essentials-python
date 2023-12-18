"""
Test P300 offline using data from an existing stream

"""

import os

from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.erp_data import ErpData
from bci_essentials.classification.erp_rg_classifier import ErpRgClassifier
from bci_essentials.utils.logger import Logger  # Logger wrapper
from saving import mne_export_erp_as_epochs

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger(name="p300_mne_plotting")

# Identify the file to simulate
filename = os.path.join("examples\\data", "p300_example.xdf")

# Choose a classifier
classifier = ErpRgClassifier()  # you can add a subset here
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)

# Set classifier settings
classifier.set_p300_clf_settings(
    n_splits=5,
    lico_expansion_factor=4,
    oversample_ratio=0,
    undersample_ratio=0,
    random_seed=35,
)

# Initialize the ERP data object
test_erp = ErpData(classifier, eeg_source, marker_source)

# Run main loop, this will do all of the classification for online or offline
test_erp.run(
    training=True,
    max_num_options=10,
    max_decisions=50,
    pp_low=0.1,
    pp_high=10,
    pp_order=5,
    plot_erp=False,
    window_start=0.0,
    window_end=0.8,
)

logger.debug("Testing mne_export_erp_as_epochs()")
mne_epochs = mne_export_erp_as_epochs(test_erp)

mne_epochs.plot(picks="eeg")

logger.debug("Ran in DEBUG mode")
