import os

from bci_essentials.eeg_data import EEG_data
from bci_essentials.erp_data import ERP_data
from bci_essentials.resting_state import get_alpha_peak, get_bandpower_features
from bci_essentials.classification.mi_classifier import MI_classifier
from bci_essentials.classification.erp_rg_classifier import ERP_rg_classifier

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "rs_example.xdf")

try:
    # Initialize data object
    test_rs = EEG_data()

    # Select a classifier
    test_rs.classifier = MI_classifier()  # you can add a subset here

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

except Exception:
    try:
        test_rs = ERP_data()

        # Choose a classifier
        test_rs.classifier = ERP_rg_classifier()  # you can add a subset here

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

    except Exception:
        print("Couldn't find resting state data")

try:
    eyes_open_windows = test_rs.eyes_open_windows
except Exception:
    print("Couldn't find eyes open data")

try:
    eyes_closed_windows = test_rs.eyes_closed_windows
except Exception:
    print("Couldn't find eyes closed data")

try:
    rest_windows = test_rs.rest_windows
except Exception:
    print("Couldn't find rest data")

fsample = test_rs.fsample
channel_labels = test_rs.channel_labels

# Get alpha peak from eyes closed?

get_alpha_peak(eyes_closed_windows, alpha_min=8, alpha_max=12, plot_psd=False)

# Get bandpower features from eyes open
abs_bandpower, rel_bandpower, rel_bandpower_mat = get_bandpower_features(
    eyes_open_windows, fs=fsample, transition_freqs=[1, 4, 8, 12, 30]
)

print(abs_bandpower[:, 0])
print(rel_bandpower[:, 0])
print(rel_bandpower_mat[:, :, 0])

print("debug")
