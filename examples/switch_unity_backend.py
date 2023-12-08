from bci_essentials.io.lsl_sources import LslEegSource, LslMarkerSource
from bci_essentials.eeg_data import EEG_data
from bci_essentials.classification.switch_mdm_classifier import Switch_mdm_classifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger()

# create LSL sources, these will block until the outlets are present
eeg_source = LslEegSource()
marker_source = LslMarkerSource()

# LETS TRY IT OUT WITH A WHOLE NEW SWITCH CLASSIFIER
classifier = Switch_mdm_classifier()
classifier.set_switch_classifier_settings(n_splits=3, rebuild=True, random_seed=35)

# Define the SWITCH data object
switch_data = EEG_data(classifier, eeg_source, marker_source)

# Run
switch_data.main(online=True, training=True)
