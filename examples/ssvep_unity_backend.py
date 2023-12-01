# import bci_essntials
from bci_essentials.eeg_data import EEG_data
from bci_essentials.classification.ssvep_riemannian_mdm_classifier import (
    SSVEP_riemannian_mdm_classifier,
)

# Define the classifier
classifier = SSVEP_riemannian_mdm_classifier()
classifier.set_ssvep_settings(
    n_splits=5, random_seed=42, n_harmonics=3, f_width=0.5, covariance_estimator="oas"
)

# Initialize the data class
test_ssvep = EEG_data()

# # Connect the streams
test_ssvep.stream_online_eeg_data()

# # Channel Selection
# initial_subset=[]
# test_ssvep.classifier.setup_channel_selection(method = "SBFS", metric="accuracy", initial_channels = initial_subset,    # wrapper setup
#                                 max_time= 999, min_channels=2, max_channels=14, performance_delta=0,      # stopping criterion
#                                 n_jobs=-1, print_output="verbose")

test_ssvep.main(
    online=True,
    training=True,
    max_samples=5120,
    pp_type="bandpass",
    pp_low=3,
    pp_high=50,
)

print("debug")
