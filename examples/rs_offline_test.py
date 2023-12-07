import os

from bci_essentials.sources.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.eeg_data import EEG_data
from bci_essentials.erp_data import ERP_data
from bci_essentials.resting_state import get_alpha_peak, get_bandpower_features
from bci_essentials.classification.mi_classifier import MI_classifier
from bci_essentials.classification.erp_rg_classifier import ERP_rg_classifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger()

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "rs_example.xdf")

try:
    # Load the xdf
    eeg_source = XdfEegSource(filename)
    marker_source = XdfMarkerSource(filename)

    # Select a classifier
    classifier = MI_classifier()  # you can add a subset here

    # Define the classifier settings
    classifier.set_mi_classifier_settings(
        n_splits=5, type="TS", random_seed=35, channel_selection="riemann"
    )

    # Initialize data object
    test_rs = EEG_data(classifier, eeg_source, marker_source)

    # Run main loop, this will do all of the classification for online or offline
    test_rs.main(
        online=False,
        training=True,
        pp_low=5,
        pp_high=30,
        pp_order=5,
    )

except Exception:
    try:
        # Load the xdf
        logger.warning("Loading file: %s", filename)
        eeg_source = XdfEegSource(filename)
        marker_source = XdfMarkerSource(filename)

        # Choose a classifier
        classifier = ERP_rg_classifier()  # you can add a subset here

        # Set classifier settings
        classifier.set_p300_clf_settings(
            n_splits=5, lico_expansion_factor=1, oversample_ratio=0, undersample_ratio=0
        )

        # Load the xdf
        test_rs = ERP_data(classifier, eeg_source, marker_source)

        # Run main loop, this will do all of the classification for online or offline
        test_rs.main(
            training=True,
            online=False,
            max_num_options=9,
            max_windows_per_option=16,
            pp_low=0.1,
            pp_high=15,
            pp_order=5,
            plot_erp=False,
            window_start=0.0,
            window_end=0.6,
        )

    except Exception:
        logger.error("Couldn't find resting state data")

try:
    eyes_open_windows = test_rs.eyes_open_windows
except Exception:
    logger.error("Couldn't find eyes open data")

try:
    eyes_closed_windows = test_rs.eyes_closed_windows
except Exception:
    logger.error("Couldn't find eyes closed data")

try:
    rest_windows = test_rs.rest_windows
except Exception:
    logger.error("Couldn't find rest data")

fsample = test_rs.fsample
channel_labels = test_rs.channel_labels

# Get alpha peak from eyes closed?
get_alpha_peak(eyes_closed_windows, alpha_min=8, alpha_max=12, plot_psd=False)

# Get bandpower features from eyes open
abs_bandpower, rel_bandpower, rel_bandpower_mat = get_bandpower_features(
    eyes_open_windows, fs=fsample, transition_freqs=[1, 4, 8, 12, 30]
)

logger.info(
    "Absolute bandpower of each band, last value is sum:\n%s", abs_bandpower[:, 0]
)

logger.info(
    "Relative bandpower of each band, last column is 1:\n%s", rel_bandpower[:, 0]
)

logger.info(
    "Matrix of band powers relative to one another:\n%s", rel_bandpower_mat[:, :, 0]
)

logger.debug("Ran in DEBUG mode")
