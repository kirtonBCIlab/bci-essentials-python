"""
Test P300 offline using data from an existing stream

"""

import os

from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.eeg_data import EegData
from bci_essentials.paradigm.p300_paradigm import P300Paradigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.erp_rg_classifier import ErpRgClassifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger(name="p300_ch_select_test")

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "p300_example.xdf")
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)

paradigm = P300Paradigm()
data_tank = DataTank()

# Choose a classifier
classifier = ErpRgClassifier()  # you can add a subset here

# Set classifier settings
classifier.set_p300_clf_settings(
    n_splits=5,
    lico_expansion_factor=1,
    oversample_ratio=0,
    undersample_ratio=0,
    random_seed=35,
    covariance_estimator="oas",
)

# Define channel selection, for SFS and SFFS you must supply atleast one initial electrode
initial_subset = []
classifier.setup_channel_selection(
    method="SBS",
    metric="accuracy",
    initial_channels=initial_subset,  # wrapper setup
    max_time=999,
    min_channels=2,
    max_channels=8,
    performance_delta=-1,  # stopping criterion
    n_jobs=-1,
    record_performance=True,
)

# Initialize the ERP data object
test_erp = EegData(classifier, eeg_source, marker_source, paradigm, data_tank)

# Run main loop, this will do all of the classification for online or offline
test_erp.setup(
    online=False,
)
test_erp.run()

logger.info("Classifier results:\n%s", classifier.results_df)
