from bci_essentials.io.lsl_sources import LslEegSource, LslMarkerSource
from bci_essentials.io.lsl_messenger import LslMessenger
from bci_essentials.eeg_data import EegData
from bci_essentials.classification.switch_mdm_classifier import SwitchMdmClassifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger()

# create LSL sources, these will block until the outlets are present
eeg_source = LslEegSource()
marker_source = LslMarkerSource()
messenger = LslMessenger()

# LETS TRY IT OUT WITH A WHOLE NEW SWITCH CLASSIFIER
classifier = SwitchMdmClassifier()
classifier.set_switch_classifier_settings(n_splits=3, rebuild=True, random_seed=35)

# Define the SWITCH data object
switch_data = EegData(classifier, eeg_source, marker_source, messenger)

# Run
switch_data.setup(online=True, training=True)
switch_data.run()
