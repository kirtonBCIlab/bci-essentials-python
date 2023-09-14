"""
Test P300 offline using data from an existing stream

"""

import os
import sys

from bci_essentials.bci_data import ERP_data
from bci_essentials.classification import ERP_rg_classifier

# mypy: disable-error-code="attr-defined"
# The above comments are for all references to ".classifier", which are not yet implemented here

# Add parent directory to path to access bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

# Initialize the ERP data object
test_erp = ERP_data()

# Choose a classifier
test_erp.classifier = ERP_rg_classifier()  # you can add a subset here

# Set classifier settings
test_erp.classifier.set_p300_clf_settings(
    n_splits=5,
    lico_expansion_factor=1,
    oversample_ratio=0,
    undersample_ratio=0,
    random_seed=35,
)

# Define channel selection
initial_subset = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "Oz", "PO8"]
test_erp.classifier.setup_channel_selection(
    method="SBFS",
    metric="accuracy",
    initial_channels=initial_subset,  # wrapper setup
    max_time=999,
    min_channels=2,
    max_channels=8,
    performance_delta=0,  # stopping criterion
    n_jobs=-1,
    print_output="verbose",
)

# Load the xdf
test_erp.load_offline_eeg_data(
    filename="examples/data/p300_example.xdf", format="xdf", print_output=False
)  # you can also add a subset here

# Run main loop, this will do all of the classification for online or offline
test_erp.main(
    training=True,
    pp_low=0.1,
    pp_high=10,
    pp_order=5,
    plot_erp=False,
    window_start=0.0,
    window_end=0.8,
    print_markers=False,
    print_training=False,
    print_fit=False,
    print_performance=True,
    print_predict=True,
)

print("debug")
