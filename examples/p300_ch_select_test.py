"""
Test P300 offline using data from an existing stream

"""

import os
import sys

# Add parent directory to path to access bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from bci_essentials.bci_data import ERP_data
from bci_essentials.classification import ERP_rg_classifier

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
    covariance_estimator="oas",
)

# Define channel selection, for SFS and SFFS you must supply atleast one initial electrode
initial_subset = []
test_erp.classifier.setup_channel_selection(
    method="SBS",
    metric="accuracy",
    initial_channels=initial_subset,  # wrapper setup
    max_time=999,
    min_channels=2,
    max_channels=8,
    performance_delta=-1,  # stopping criterion
    n_jobs=-1,
    print_output="verbose",
    record_performance=True,
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

print(test_erp.classifier.results_df)

print("debug")
