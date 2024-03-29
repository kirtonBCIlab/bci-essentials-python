"""
Test Motor Imagery (MI) classification offline using data from an existing stream

"""

# Stock libraries
import os


# bci_essentials
from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.eeg_data import EegData
from bci_essentials.classification.mi_classifier import MiClassifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Create a logger for the user script
# Note: This is separate from the bci_essentials logger and does not
# modify the logging behaviour of the bci_essentials package
logger = Logger(name="mi_ch_select_test")
# Set the logging level of the user script logger to DEBUG
logger.setLevel(Logger.DEBUG)

# Modify the logging behaviour of the bci_essentials package to DEBUG
bessy_logger = Logger()  # bci_essentials logger
bessy_logger.setLevel(Logger.DEBUG)

logger.debug("Starting user script in DEBUG mode")

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "mi_example_2.xdf")
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)

# Select a classifier
classifier = MiClassifier()  # you can add a subset here

# Define the classifier settings
classifier.set_mi_classifier_settings(n_splits=5, type="TS", random_seed=35)

# Define channel selection settings
initial_subset = []
classifier.setup_channel_selection(
    method="SFFS",
    metric="accuracy",
    iterative_selection=True,
    initial_channels=initial_subset,  # wrapper setup
    max_time=4,
    min_channels=0,
    max_channels=20,
    performance_delta=-0.05,  # stopping criterion
    n_jobs=-1,
    record_performance=True,
)

# Initialize data object
test_mi = EegData(classifier, eeg_source, marker_source)

# Run main loop, this will do all of the classification for online or offline
test_mi.setup(
    online=False,
    training=True,
    pp_low=5,
    pp_high=30,
    pp_order=5,
)
test_mi.run()

logger.info("%s", classifier.results_df)

logger.debug("Finished running in DEBUG mode")
