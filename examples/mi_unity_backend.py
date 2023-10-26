from bci_essentials.bci_data import EEG_data
from bci_essentials.classification.mi_classifier import MI_classifier

# Define the MI data object
mi_data = EEG_data()

# Select a classifier
mi_data.classifier = MI_classifier()  # you can add a subset here

# Set settings
mi_data.classifier.set_mi_classifier_settings(n_splits=3, type="TS", random_seed=35)

# Connect the streams

mi_data.stream_online_eeg_data()  # you can also add a subset here


# Run
mi_data.main(online=True, training=True)
