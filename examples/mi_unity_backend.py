from bci_essentials.io.lsl_sources import LslEegSource, LslMarkerSource
from bci_essentials.io.lsl_messenger import LslMessenger
from bci_essentials.eeg_data import EegData
from bci_essentials.classification.mi_classifier import MiClassifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger()

# create LSL sources, these will block until the outlets are present
eeg_source = LslEegSource()
marker_source = LslMarkerSource()
messenger = LslMessenger()

# Select a classifier
classifier = MiClassifier()  # you can add a subset here

# Set settings
classifier.set_mi_classifier_settings(n_splits=3, type="TS", random_seed=35)

# Define the MI data object
mi_data = EegData(classifier, eeg_source, marker_source, messenger)

# Run
mi_data.main(online=True, training=True)
