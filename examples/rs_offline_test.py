import os
import sys


# Add parent directory to path to access bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

# from src.bci_data import *
from bci_essentials.bci_data import *
from bci_essentials.visuals import *
from bci_essentials.resting_state import *

filename = "examples/data/rs_example.xdf"

try:
    # Initialize data object
    test_rs = EEG_data()

    # Select a classifier
    test_rs.classifier = mi_classifier()  # you can add a subset here

    # Define the classifier settings
    test_rs.classifier.set_mi_classifier_settings(
        n_splits=5, type="TS", random_seed=35, channel_selection="riemann"
    )

    # Load the xdf
    test_rs.load_offline_eeg_data(
        filename=filename, print_output=False
    )  # you can also add a subset here

    # Run main loop, this will do all of the classification for online or offline
    test_rs.main(
        online=False,
        training=True,
        pp_low=5,
        pp_high=30,
        pp_order=5,
        print_markers=False,
        print_training=False,
        print_fit=False,
        print_performance=False,
        print_predict=False,
    )

except:
    try:
        test_rs = ERP_data()

        # Choose a classifier
        test_rs.classifier = erp_rg_classifier()  # you can add a subset here

        # Set classifier settings
        test_rs.classifier.set_p300_clf_settings(
            n_splits=5, lico_expansion_factor=1, oversample_ratio=0, undersample_ratio=0
        )

        # Load the xdf
        print(filename)
        test_rs.load_offline_eeg_data(
            filename=filename, format="xdf", print_output=False
        )  # you can also add a subset here

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
            print_markers=False,
            print_training=False,
            print_fit=False,
            print_performance=False,
            print_predict=False,
        )

    except:
        print("Couldn't find resting state data")

try:
    eyes_open_windows = test_rs.eyes_open_windows
except:
    print("Couldn't find eyes open data")

try:
    eyes_closed_windows = test_rs.eyes_closed_windows
except:
    print("Couldn't find eyes closed data")

try:
    rest_windows = test_rs.rest_windows
except:
    print("Couldn't find rest data")

fsample = test_rs.fsample
channel_labels = test_rs.channel_labels

# Get alpha peak from eyes closed?

get_alpha_peak(eyes_closed_windows, alpha_min = 8, alpha_max = 12, plot_psd = False)

# Get bandpower features from eyes open
abs_bandpower, rel_bandpower, rel_bandpower_mat = get_bandpower_features(
    eyes_open_windows, fs=fsample, transition_freqs=[1, 4, 8, 12, 30]
)

print(abs_bandpower[:,0])
print(rel_bandpower[:,0])
print(rel_bandpower_mat[:,:,0])

print("debug")
