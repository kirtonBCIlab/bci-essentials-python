import unittest
import os

from bci_essentials.sources.xdf_sources import XdfMarkerSource, XdfEegSource
from bci_essentials.eeg_data import EEG_data
from bci_essentials.erp_data import ERP_data
from bci_essentials.classification.mi_classifier import MI_classifier
from bci_essentials.classification.erp_rg_classifier import ERP_rg_classifier
from bci_essentials.classification.ssvep_riemannian_mdm_classifier import (
    SSVEP_riemannian_mdm_classifier,
)

data_folder_path = os.path.join("examples", "data")


class TestSmoke(unittest.TestCase):
    def test_mi_offline(self):
        # Get the MI example data from ./examples/data
        xdf_path = os.path.join(data_folder_path, "mi_example.xdf")
        eeg_source = XdfEegSource(xdf_path)
        marker_source = XdfMarkerSource(xdf_path)

        # Select a classifier
        classifier = MI_classifier()
        classifier.set_mi_classifier_settings(
            n_splits=5, type="TS", random_seed=35, channel_selection="riemann"
        )

        # Load the data
        data = EEG_data(classifier, eeg_source, marker_source)

        # Run main loop, this will do all of the classification for online or offline
        data.main(
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

        # Check that a model was trained
        self.assertIsNotNone(classifier.clf)

        # Check that accuracy, precision, recall and covariance matrix exist
        self.assertIsNotNone(classifier.offline_accuracy)
        self.assertIsNotNone(classifier.offline_precision)
        self.assertIsNotNone(classifier.offline_recall)
        self.assertIsNotNone(classifier.offline_cm)

        # Check that the list of class predictions exists
        self.assertIsNotNone(classifier.predictions)

        print("MI test complete")

    def test_p300_offline(self):
        # Get the P300 example data from ./examples/data
        xdf_path = os.path.join(data_folder_path, "p300_example.xdf")
        eeg_source = XdfEegSource(xdf_path)
        marker_source = XdfMarkerSource(xdf_path)

        # Select a classifier
        classifier = ERP_rg_classifier()
        classifier.set_p300_clf_settings(
            n_splits=5,
            lico_expansion_factor=4,
            oversample_ratio=0,
            undersample_ratio=0,
            random_seed=35,
        )

        # Load the data
        data = ERP_data(classifier, eeg_source, marker_source)

        # Run main loop, this will do all of the classification for online or offline
        data.main(
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

        # Check that a model was trained
        self.assertIsNotNone(classifier.clf)

        # Check that accuracy, precision, recall and covariance matrix exist
        self.assertIsNotNone(classifier.offline_accuracy)
        self.assertIsNotNone(classifier.offline_precision)
        self.assertIsNotNone(classifier.offline_recall)
        self.assertIsNotNone(classifier.offline_cm)

        # Check that the list of class predictions exists
        self.assertIsNotNone(classifier.predictions)

        print("P300 test complete")

    def test_ssvep_offline(self):
        # Get the SSVEP example data from ./examples/data
        xdf_path = os.path.join(data_folder_path, "ssvep_example.xdf")
        eeg_source = XdfEegSource(xdf_path)
        marker_source = XdfMarkerSource(xdf_path)

        # Select a classifier
        classifier = SSVEP_riemannian_mdm_classifier()
        classifier.set_ssvep_settings(
            n_splits=3,
            random_seed=35,
            n_harmonics=2,
            f_width=0.2,
            covariance_estimator="scm",
        )

        # Load the data
        data = EEG_data(classifier, eeg_source, marker_source)

        # Run main loop, this will do all of the classification for online or offline
        data.main(
            online=False,
            training=True,
            pp_type="bandpass",
            pp_low=3,
            pp_high=50,
            pp_order=5,
            print_markers=False,
            print_training=False,
            print_fit=False,
            print_performance=False,
            print_predict=False,
        )

        # Check that a model was trained
        self.assertIsNotNone(classifier.clf)

        # Check that accuracy, precision, recall and covariance matrix exist
        self.assertIsNotNone(classifier.offline_accuracy)
        self.assertIsNotNone(classifier.offline_precision)
        self.assertIsNotNone(classifier.offline_recall)
        self.assertIsNotNone(classifier.offline_cm)

        # Check that the list of class predictions exists
        self.assertIsNotNone(classifier.predictions)

        print("SSVEP test complete")


if __name__ == "__main__":
    unittest.main()
