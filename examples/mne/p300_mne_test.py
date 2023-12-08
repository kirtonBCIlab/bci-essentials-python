"""
Test P300 offline using data from an existing stream

"""

from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.erp_data import ERP_data
from bci_essentials.classification.erp_rg_classifier import ERP_rg_classifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger()

# Identify the file to simulate
# This won't work on anyone else's computer
filename = "C:\\Users\\brian\\OneDrive\\Documents\\BCI\\BCIEssentials\\fatigueDataAnalysis\\fatigueData\\participants\\sub-P06_p300\\ses-postRS_p300\\eeg\\sub-P06_p300_ses-postRS_p300_task-T1_run-001_eeg.xdf"

# Choose a classifier
classifier = ERP_rg_classifier()  # you can add a subset here
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)

# Set classifier settings
classifier.set_p300_clf_settings(
    n_splits=5,
    lico_expansion_factor=4,
    oversample_ratio=0,
    undersample_ratio=0,
    random_seed=35,
)

# Initialize the ERP data object
test_erp = ERP_data(classifier, eeg_source, marker_source)

# Run main loop, this will do all of the classification for online or offline
test_erp.main(
    training=True,
    max_num_options=10,
    max_decisions=50,
    pp_low=0.1,
    pp_high=10,
    pp_order=5,
    plot_erp=False,
    window_start=0.0,
    window_end=0.8,
)


logger.debug("test_erp.mne_export_resting_state_as_raw()")
rs_mne = test_erp.mne_export_resting_state_as_raw()

logger.debug("test_erp.mne_export_as_epochs()")
mne_epochs = test_erp.mne_export_as_epochs()

mne_epochs.plot(picks="eeg")

logger.debug("Ran in DEBUG mode")
