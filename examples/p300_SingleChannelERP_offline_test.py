"""
Test P300 offline using data from an existing stream

"""

import os

from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.erp_data import ErpData
from bci_essentials.classification.erp_single_channel_classifier import (
    ErpSingleChannelClassifier,
)

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "p300_example.xdf")
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)

# Choose a classifier
classifier = ErpSingleChannelClassifier()

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
test_erp.setup(
    training=True,
    pp_low=0.1,
    pp_high=10,
    pp_order=5,
    plot_erp=False,
    trial_start=0.0,
    trial_end=0.8,
)
test_erp.run()
