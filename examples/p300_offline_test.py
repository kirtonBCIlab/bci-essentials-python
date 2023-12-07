"""
Test P300 offline using data from an existing stream

"""

import os

from bci_essentials.sources.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.erp_data import ERP_data
from bci_essentials.classification.erp_rg_classifier import ERP_rg_classifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger at the default level of logging.INFO
logger = Logger()

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "p300_example.xdf")
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)

# Choose a classifier
classifier = ERP_rg_classifier()  # you can add a subset here

# Set classifier settings
classifier.set_p300_clf_settings(
    n_splits=5,
    lico_expansion_factor=4,
    oversample_ratio=0,
    undersample_ratio=0,
    random_seed=35,
    covariance_estimator="oas",
)

# Initialize the ERP data object
test_erp = ERP_data(classifier, eeg_source, marker_source)

# Run main loop, this will do all of the classification for online or offline
test_erp.main(
    training=True,
    pp_low=0.1,
    pp_high=10,
    pp_order=5,
    plot_erp=False,
    window_start=0.0,
    window_end=0.8,
)

logger.debug("Finished running in DEBUG mode")
