from ..bci_essentials.bci_data import EEG_data
from ..bci_essentials.classification.ssvep_basic_tf_classifier \
    import SSVEP_basic_tf_classifier

# Initialize the EEG Data
test_ssvep = EEG_data()

# set train complete to true so that predictions will be allowed
test_ssvep.train_complete = True

# Define the classifier
test_ssvep.classifier = SSVEP_basic_tf_classifier()
target_freqs = [9, 9.6, 10.28, 11.07, 12, 13.09, 14.4]

# Connect the streams
test_ssvep.stream_online_eeg_data()

# Run
test_ssvep.main(online=True, training=False, train_complete=True)
