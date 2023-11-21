import unittest
import os
import numpy as np

from bci_essentials.eeg_data import EEG_data
from bci_essentials.erp_data import ERP_data
from bci_essentials.classification.mi_classifier import MI_classifier
from bci_essentials.classification.erp_rg_classifier import ERP_rg_classifier
from bci_essentials.session_saving import save_classifier, load_classifier

data_folder_path = os.path.join("examples", "data")


class TestClassifierSave(unittest.TestCase):
    def test_mi_manual_classifier_save(self):
        # Create an EEG_data object
        mi_data1 = EEG_data()

        # Get the MI example data from ./examples/data
        mi_xdf_path = os.path.join(data_folder_path, "mi_example.xdf")

        # Load the data
        mi_data1 = EEG_data()
        mi_data1.load_offline_eeg_data(filename=mi_xdf_path, print_output=False)

        # Select a classifier
        mi_data1.classifier = MI_classifier()

        # Define the classifier settings
        mi_data1.classifier.set_mi_classifier_settings(
            n_splits=5,
            type="TS",
            random_seed=35,
            channel_selection="riemann",
            covariance_estimator="oas",
        )

        # Run main loop, this will do all of the classification for online or offline
        mi_data1.main(
            online=False,
            training=True,
            pp_low=5,
            pp_high=50,
            pp_order=5,
            print_markers=False,
            print_training=False,
            print_fit=False,
            print_performance=False,
            print_predict=False,
        )

        # Save the classifier model
        save_classifier(mi_data1.classifier, "test_mi_classifier.pkl")

        # Create a new EEG_data object
        mi_data2 = EEG_data()

        # Load the data
        mi_data2.load_offline_eeg_data(filename=mi_xdf_path, print_output=False)

        # Select a classifier
        mi_data2.classifier = load_classifier("test_mi_classifier.pkl")

        # Check that all values of the classifier's numpy arrays are the same
        self.assertTrue(np.array_equal(mi_data1.classifier.X, mi_data2.classifier.X))

        print("Classifier reloaded")

        # Delete classifier predictions
        mi_data2.classifier.predictions = []

        # Run the main loop
        mi_data2.main(
            online=False,
            training=False,
            train_complete=True,
            pp_low=5,
            pp_high=50,
            pp_order=5,
            print_markers=False,
            print_training=False,
            print_fit=False,
            print_performance=False,
            print_predict=False,
        )

        # Check that X and y are the same
        self.assertTrue(np.array_equal(mi_data1.classifier.X, mi_data2.classifier.X))
        self.assertTrue(np.array_equal(mi_data1.classifier.y, mi_data2.classifier.y))

        # Check that the last predictions are the same for the mi_example given, the last 78 predictions should be the same
        self.assertTrue(
            np.array_equal(
                mi_data1.classifier.predictions[-78:],
                mi_data2.classifier.predictions[-78:],
            )
        )

        print("MI classifier save/load test passed")

    def test_p300_manual_classifier_save(self):
        # Get the P300 example data from ./examples/data
        p300_xdf_path = os.path.join(data_folder_path, "p300_example.xdf")

        # Load the data
        p300_data1 = ERP_data()
        p300_data1.load_offline_eeg_data(filename=p300_xdf_path, print_output=False)

        # Select a classifier
        p300_data1.classifier = ERP_rg_classifier()

        # Define the classifier settings
        p300_data1.classifier.set_p300_clf_settings(
            n_splits=5,
            lico_expansion_factor=4,
            oversample_ratio=0,
            undersample_ratio=0,
            random_seed=35,
        )

        # Run main loop, this will do all of the classification for online or offline
        p300_data1.main(
            online=False,
            training=True,
            pp_low=0.1,
            pp_high=10,
            pp_order=5,
            plot_erp=False,
            window_start=0.0,
            window_end=0.8,
            max_num_options=9,
            max_windows_per_option=16,
            max_decisions=20,
            print_markers=False,
            print_training=False,
            print_fit=False,
            print_performance=False,
            print_predict=False,
        )

        # Save the classifier model
        save_classifier(p300_data1.classifier, "test_p300_classifier.pkl")

        # Create a new ERP_data object
        p300_data2 = ERP_data()

        # Load the data
        p300_data2.load_offline_eeg_data(filename=p300_xdf_path, print_output=False)

        # Load the classifier
        p300_data2.classifier = load_classifier("test_p300_classifier.pkl")

        # Check that all values of the classifier's numpy arrays are the same
        self.assertTrue(
            np.array_equal(p300_data1.classifier.X, p300_data2.classifier.X)
        )

        print("Classifier reloaded")

        # Delete classifier predictions
        p300_data2.classifier.predictions = []

        # Run the main loop
        p300_data2.main(
            online=False,
            training=False,
            train_complete=True,
            pp_low=0.1,
            pp_high=10,
            pp_order=5,
            plot_erp=False,
            window_start=0.0,
            window_end=0.8,
            max_num_options=9,
            max_windows_per_option=16,
            max_decisions=20,
            print_markers=False,
            print_training=False,
            print_fit=False,
            print_performance=False,
            print_predict=False,
        )

        # Check that X and y are the same
        self.assertTrue(
            np.array_equal(p300_data1.classifier.X, p300_data2.classifier.X)
        )
        self.assertTrue(
            np.array_equal(p300_data1.classifier.y, p300_data2.classifier.y)
        )

        # Check that the last predictions are the same for the p300_example given, the last predictions should be the same
        num_preds = len(p300_data1.classifier.predictions)
        self.assertTrue(
            np.array_equal(
                p300_data1.classifier.predictions[-num_preds:],
                p300_data2.classifier.predictions[-num_preds:],
            )
        )


if __name__ == "__main__":
    unittest.main()
