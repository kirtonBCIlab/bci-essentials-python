from bci_essentials.sources.lsl_sources import LslEegSource, LslMarkerSource
from bci_essentials.eeg_data import EEG_data
from bci_essentials.classification.mi_classifier import MI_classifier

# create LSL sources, these will block until the outlets are present
eeg_source = LslEegSource()
marker_source = LslMarkerSource()

# Select a classifier
classifier = MI_classifier()  # you can add a subset here

# Set settings
classifier.set_mi_classifier_settings(n_splits=3, type="TS", random_seed=35)

# Define the MI data object
mi_data = EEG_data(eeg_source, marker_source, classifier)

# Run
mi_data.main(online=True, training=True)
