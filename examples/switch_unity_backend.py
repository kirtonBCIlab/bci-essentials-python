from bci_essentials.bci_data import EEG_data
from bci_essentials.classification.switch_mdm_classifier import Switch_mdm_classifier

# Define the SWITCH data object
switch_data = EEG_data()

# LETS TRY IT OUT WITH A WHOLE NEW SWITCH CLASSIFIER
switch_data.classifier = Switch_mdm_classifier()

switch_data.classifier.set_switch_classifier_settings(
    n_splits=3, rebuild=True, random_seed=35
)
# Connect the streams
switch_data.stream_online_eeg_data()

# Run
switch_data.main(online=True, training=True)
