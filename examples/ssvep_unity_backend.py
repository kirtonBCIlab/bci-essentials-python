from bci_essentials.sources.lsl_sources import LslEegSource, LslMarkerSource
from bci_essentials.eeg_data import EEG_data
from bci_essentials.classification.ssvep_riemannian_mdm_classifier import (
    SSVEP_riemannian_mdm_classifier,
)
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger()

# create LSL sources, these will block until the outlets are present
eeg_source = LslEegSource()
marker_source = LslMarkerSource()

# Define the classifier
classifier = SSVEP_riemannian_mdm_classifier()
classifier.set_ssvep_settings(
    n_splits=5, random_seed=42, n_harmonics=3, f_width=0.5, covariance_estimator="oas"
)

# Initialize the data class
test_ssvep = EEG_data(classifier, eeg_source, marker_source)

# # Channel Selection
# initial_subset=[]
# test_ssvep.classifier.setup_channel_selection(method = "SBFS", metric="accuracy", initial_channels = initial_subset,    # wrapper setup
#                                 max_time= 999, min_channels=2, max_channels=14, performance_delta=0,      # stopping criterion
#                                 n_jobs=-1)

test_ssvep.main(
    online=True,
    training=True,
    max_samples=5120,
    pp_type="bandpass",
    pp_low=3,
    pp_high=50,
)

logger.debug("Ran in DEBUG mode")
