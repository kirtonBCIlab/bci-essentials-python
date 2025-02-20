import os

from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.bci_controller import BciController
from bci_essentials.paradigm.mi_paradigm import MiParadigm
from bci_essentials.data_tank.data_tank import DataTank

# from bci_essentials.erp_data import ErpData
from bci_essentials.resting_state import get_alpha_peak, get_bandpower_features
from bci_essentials.classification.mi_classifier import MiClassifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger(name="rs_offline_test")

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "rs_example.xdf")


# Load the xdf
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)

paradigm = MiParadigm(live_update=True, iterative_training=True)
data_tank = DataTank()

# Select a classifier
classifier = MiClassifier()  # you can add a subset here

# Define the classifier settings
classifier.set_mi_classifier_settings(
    n_splits=5, type="TS", random_seed=35
)

# Initialize data object
test_rs = BciController(classifier, eeg_source, marker_source, paradigm, data_tank)

# Run main loop, this will do all of the classification for online or offline
test_rs.setup(
    online=False,
)
test_rs.run()

try:
    eyes_open_trials = data_tank.resting_state_data["eyes_open_trials"]
except Exception:
    logger.error("Couldn't find eyes open data")

try:
    eyes_closed_trials = data_tank.resting_state_data["eyes_closed_trials"]
except Exception:
    logger.error("Couldn't find eyes closed data")

try:
    rest_trials = data_tank.resting_state_data["rest_trials"]
except Exception:
    logger.error("Couldn't find rest data")

fsample = test_rs.fsample
channel_labels = test_rs.channel_labels

# Get alpha peak from eyes closed?
get_alpha_peak(eyes_closed_trials, alpha_min=8, alpha_max=12, plot_psd=False)

# Get bandpower features from eyes open
abs_bandpower, rel_bandpower, rel_bandpower_mat = get_bandpower_features(
    eyes_open_trials, fs=fsample, transition_freqs=[1, 4, 8, 12, 30]
)

logger.info(
    "Absolute bandpower of each band, last value is sum:\n%s", abs_bandpower[:, 0]
)

logger.info(
    "Relative bandpower of each band, last column is 1:\n%s", rel_bandpower[:, 0]
)

logger.info(
    "Matrix of band powers relative to one another:\n%s", rel_bandpower_mat[:, :, 0]
)

logger.debug("Ran in DEBUG mode")
