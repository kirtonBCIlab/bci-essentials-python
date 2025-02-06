"""
Test Motor Imagery (MI) classification offline using data from an existing stream

"""

# Stock libraries
import os

# bci_essentials
from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.bci_controller import BciController
from bci_essentials.paradigm.mi_paradigm import MiParadigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.mi_classifier import MiClassifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger at the default level of logging.INFO
logger = Logger(name="mi_ch_select_test")

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
classifier.set_mi_classifier_settings(n_splits=5, type="TS", random_seed=35)

# Define channel selection settings
initial_subset = []
classifier.setup_channel_selection(
    method="SBFS",
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
test_mi = BciController(classifier, eeg_source, marker_source, paradigm, data_tank)

# Run main loop, this will do all of the classification for online or offline
test_mi.setup(
    online=False,
)
test_mi.run()

logger.info("%s", classifier.results_df)
