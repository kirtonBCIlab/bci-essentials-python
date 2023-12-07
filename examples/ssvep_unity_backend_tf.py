from bci_essentials.sources.lsl_sources import LslEegSource, LslMarkerSource
from bci_essentials.eeg_data import EEG_data
from bci_essentials.classification.ssvep_basic_tf_classifier import (
    SSVEP_basic_tf_classifier,
)
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger()

# create LSL sources, these will block until the outlets are present
eeg_source = LslEegSource()
marker_source = LslMarkerSource()

# Define the classifier
classifier = SSVEP_basic_tf_classifier()

# Initialize the EEG Data
test_ssvep = EEG_data(classifier, eeg_source, marker_source)

# set train complete to true so that predictions will be allowed
test_ssvep.train_complete = True

target_freqs = [9, 9.6, 10.28, 11.07, 12, 13.09, 14.4]

# Run
test_ssvep.main(online=True, training=False, train_complete=True)
