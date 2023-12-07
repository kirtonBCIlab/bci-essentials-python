"""
Test Motor Imagery (MI) classification offline using data from an existing stream

"""
# Stock libraries
import os

# bci_essentials
from bci_essentials.sources.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.eeg_data import EEG_data
from bci_essentials.classification.mi_classifier import MI_classifier

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "mi_example_2.xdf")
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)

# Select a classifier
classifier = MI_classifier()  # you can add a subset here

# Define the classifier settings
classifier.set_mi_classifier_settings(n_splits=5, type="TS", random_seed=35)

# Define channel selection settings
initial_subset = []
classifier.setup_channel_selection(
    method="SFFS",
    metric="accuracy",
    iterative_selection=True,
    initial_channels=initial_subset,  # wrapper setup
    max_time=4,
    min_channels=0,
    max_channels=20,
    performance_delta=-0.05,  # stopping criterion
    n_jobs=-1,
    print_output="silent",
    record_performance=True,
)

# Initialize data object
test_mi = EEG_data(classifier, eeg_source, marker_source)

# Run main loop, this will do all of the classification for online or offline
test_mi.main(
    online=False,
    training=True,
    pp_low=5,
    pp_high=30,
    pp_order=5,
    print_markers=True,
    print_training=False,
    print_fit=False,
    print_performance=True,
    print_predict=False,
)

print(classifier.results_df)

print("debug")
