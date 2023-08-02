import unittest
import os
import sys

from bci_essentials.bci_data import EEG_data, ERP_data
from bci_essentials.classification import (
    MI_classifier,
    ERP_rg_classifier,
    SSVEP_riemannian_mdm_classifier,
)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


class TestLoadData(unittest.TestCase):
    def test_mi_offline(self):
        # Get the MI example data from /examples/data
        mi_xdf_path = "examples//data//mi_example.xdf"

        # Load the data
        mi_data = EEG_data()
        mi_data.load_offline_eeg_data(filename=mi_xdf_path, print_output=False)

        # Select a classifier
        mi_data.classifier = MI_classifier()

        # Define the classifier settings
        mi_data.classifier.set_mi_classifier_settings(
            n_splits=5, type="TS", random_seed=35, channel_selection="riemann"
        )

        # Run main loop, this will do all of the classification for online or offline
        mi_data.main(
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
        self.assertIsNotNone(mi_data.classifier.clf)

        # Check that accuracy, precision, recall and covariance matrix exist
        self.assertIsNotNone(mi_data.classifier.offline_accuracy)
        self.assertIsNotNone(mi_data.classifier.offline_precision)
        self.assertIsNotNone(mi_data.classifier.offline_recall)
        self.assertIsNotNone(mi_data.classifier.offline_cm)

        # Check that the list of class predictions exists
        self.assertIsNotNone(mi_data.classifier.predictions)

    def test_p300_offline(self):
        # Get the P300 example data from /examples/data
        p300_xdf_path = "examples//data//p300_example.xdf"

        # Load the data
        p300_data = ERP_data()
        p300_data.load_offline_eeg_data(filename=p300_xdf_path, print_output=False)

        # Select a classifier
        p300_data.classifier = ERP_rg_classifier()

        # Define the classifier settings
        p300_data.classifier.set_p300_clf_settings(
            n_splits=5,
            lico_expansion_factor=4,
            oversample_ratio=0,
            undersample_ratio=0,
            random_seed=35,
        )

        # Run main loop, this will do all of the classification for online or offline
        p300_data.main(
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
        self.assertIsNotNone(p300_data.classifier.clf)

        # Check that accuracy, precision, recall and covariance matrix exist
        self.assertIsNotNone(p300_data.classifier.offline_accuracy)
        self.assertIsNotNone(p300_data.classifier.offline_precision)
        self.assertIsNotNone(p300_data.classifier.offline_recall)
        self.assertIsNotNone(p300_data.classifier.offline_cm)

        # Check that the list of class predictions exists
        self.assertIsNotNone(p300_data.classifier.predictions)

    def test_ssvep_offline(self):
        # Get the SSVEP example data from /examples/data
        ssvep_xdf_path = "examples//data//ssvep_example.xdf"

        # Load the data
        ssvep_data = EEG_data()
        ssvep_data.load_offline_eeg_data(filename=ssvep_xdf_path, print_output=False)

        # Select a classifier
        ssvep_data.classifier = SSVEP_riemannian_mdm_classifier()

        # Define the classifier settings
        ssvep_data.classifier.set_ssvep_settings(
            n_splits=3,
            random_seed=35,
            n_harmonics=2,
            f_width=0.2,
            covariance_estimator="scm",
        )

        # Run main loop, this will do all of the classification for online or offline
        ssvep_data.main(
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
        self.assertIsNotNone(ssvep_data.classifier.clf)

        # Check that accuracy, precision, recall and covariance matrix exist
        self.assertIsNotNone(ssvep_data.classifier.offline_accuracy)
        self.assertIsNotNone(ssvep_data.classifier.offline_precision)
        self.assertIsNotNone(ssvep_data.classifier.offline_recall)
        self.assertIsNotNone(ssvep_data.classifier.offline_cm)

        # Check that the list of class predictions exists
        self.assertIsNotNone(ssvep_data.classifier.predictions)


if __name__ == "__main__":
    unittest.main()
