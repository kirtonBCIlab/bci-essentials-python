import unittest
import os
import numpy as np

from bci_essentials.sources.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.eeg_data import EEG_data
from bci_essentials.erp_data import ERP_data
from bci_essentials.classification.mi_classifier import MI_classifier
from bci_essentials.classification.erp_rg_classifier import ERP_rg_classifier
from bci_essentials.session_saving import save_classifier, load_classifier

data_folder_path = os.path.join("examples", "data")


class TestClassifierSave(unittest.TestCase):
    def test_mi_manual_classifier_save(self):
        # Get the MI example data from ./examples/data
        xdf_path = os.path.join(data_folder_path, "mi_example.xdf")
        eeg_source1 = XdfEegSource(xdf_path)
        marker_source1 = XdfMarkerSource(xdf_path)

        # Select a classifier
        classifier1 = MI_classifier()
        classifier1.set_mi_classifier_settings(
            n_splits=5,
            type="TS",
            random_seed=35,
            channel_selection="riemann",
            covariance_estimator="oas",
        )

        # Load the data
        data1 = EEG_data(classifier1, eeg_source1, marker_source1)

        # Run main loop, this will do all of the classification for online or offline
        data1.main(
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

        # Save the classifier
        save_classifier(classifier1, "test_mi_classifier.pkl")

        # Load the classifier
        classifier2 = load_classifier("test_mi_classifier.pkl")

        # Create a new EEG_data object, recreate xdf sources to reload files
        eeg_source2 = XdfEegSource(xdf_path)
        marker_source2 = XdfMarkerSource(xdf_path)
        data2 = EEG_data(classifier2, eeg_source2, marker_source2)

        # Check that all values of the classifier's numpy arrays are the same
        self.assertTrue(np.array_equal(classifier1.X, classifier2.X))

        print("Classifier reloaded")

        # Delete classifier predictions
        classifier2.predictions = []

        # Run the main loop
        data2.main(
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
        self.assertTrue(np.array_equal(classifier1.X, classifier2.X))
        self.assertTrue(np.array_equal(classifier1.y, classifier2.y))

        # Check that the last predictions are the same for the mi_example given, the last 78 predictions should be the same
        self.assertTrue(
            np.array_equal(
                classifier1.predictions[-78:],
                classifier2.predictions[-78:],
            )
        )

        print("MI classifier save/load test passed")

    def test_p300_manual_classifier_save(self):
        # Get the P300 example data from ./examples/data
        xdf_path = os.path.join(data_folder_path, "p300_example.xdf")
        eeg_source1 = XdfEegSource(xdf_path)
        marker_source1 = XdfMarkerSource(xdf_path)

        # Select a classifier
        classifier1 = ERP_rg_classifier()
        classifier1.set_p300_clf_settings(
            n_splits=5,
            lico_expansion_factor=4,
            oversample_ratio=0,
            undersample_ratio=0,
            random_seed=35,
        )

        # Load the data
        data1 = ERP_data(classifier1, eeg_source1, marker_source1)

        # Run main loop, this will do all of the classification for online or offline
        data1.main(
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
        save_classifier(classifier1, "test_p300_classifier.pkl")

        # Load the classifier
        classifier2 = load_classifier("test_p300_classifier.pkl")

        # Create a new ERP_data object, recreate xdf sources to reload files
        eeg_source2 = XdfEegSource(xdf_path)
        marker_source2 = XdfMarkerSource(xdf_path)
        data2 = ERP_data(classifier2, eeg_source2, marker_source2)

        # Check that all values of the classifier's numpy arrays are the same
        self.assertTrue(np.array_equal(classifier1.X, classifier2.X))

        print("Classifier reloaded")

        # Delete classifier predictions
        classifier2.predictions = []

        # Run the main loop
        data2.main(
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
        self.assertTrue(np.array_equal(classifier1.X, classifier2.X))
        self.assertTrue(np.array_equal(classifier1.y, classifier2.y))

        # Check that the last predictions are the same for the p300_example given, the last predictions should be the same
        num_preds = len(classifier1.predictions)
        self.assertTrue(
            np.array_equal(
                classifier1.predictions[-num_preds:],
                classifier2.predictions[-num_preds:],
            )
        )


if __name__ == "__main__":
    unittest.main()
