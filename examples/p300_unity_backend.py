
import os
import sys

# # Add parent directory to path to access bci_essentials
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))

from bci_essentials.bci_data import *

# Initialize the ERP
test_erp = ERP_data()

# Set classifier settings ()
test_erp.classifier = erp_rg_classifier() # you can add a subset here

# Set some settings
test_erp.classifier.set_p300_clf_settings(n_splits=5, lico_expansion_factor=1, oversample_ratio=1, undersample_ratio=0)

# Connect the streams
test_erp.stream_online_eeg_data() # you can also add a subset here

# Run main
test_erp.main(online=True, training=True, pp_low=0.1, pp_high=15, pp_order=5, plot_erp=False, window_start=0.0, window_end=0.6)