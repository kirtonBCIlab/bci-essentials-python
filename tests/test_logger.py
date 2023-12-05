import unittest
import os
import logging

from bci_essentials.eeg_data import EEG_data
from bci_essentials.erp_data import ERP_data
from bci_essentials.classification.mi_classifier import MI_classifier
from bci_essentials.classification.erp_rg_classifier import ERP_rg_classifier
from bci_essentials.classification.ssvep_riemannian_mdm_classifier import (
    SSVEP_riemannian_mdm_classifier,
)
from bci_essentials.utils.logger import Logger  # Logger class

# data_folder_path = os.path.join("examples", "data")
data_folder_path = os.path.join("../examples", "data")

# Instantiate a logger for the module at the level of logging.debug
logger = Logger(name=__name__,level=logging.DEBUG)
logger.debug("Starting test_logger.py...")


class TestSmoke(unittest.TestCase):
    def test_mi_offline(self):
        # Get the MI example data from ./examples/data
        mi_xdf_path = os.path.join(data_folder_path, "mi_example.xdf")

        # Select a classifier
        classifier = MI_classifier()
        classifier.set_mi_classifier_settings(
            n_splits=5, type="TS", random_seed=35, channel_selection="riemann"
        )

        # Load the data
        data = EEG_data(classifier)
        data.load_offline_eeg_data(filename=mi_xdf_path, print_output=True)

        # Run main loop, this will do all of the classification for online or offline
        data.main(
            online=False,
            training=True,
            pp_low=5,
            pp_high=50,
            pp_order=5,
            print_markers=True,
            print_training=True,
            print_fit=True,
            print_performance=True,
            print_predict=True,
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

        logger.info("MI test complete")

    def test_p300_offline(self):
        # Get the P300 example data from ./examples/data
        p300_xdf_path = os.path.join(data_folder_path, "p300_example.xdf")

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
        data = ERP_data(classifier)
        data.load_offline_eeg_data(filename=p300_xdf_path, print_output=True)

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
            print_markers=True,
            print_training=True,
            print_fit=True,
            print_performance=True,
            print_predict=True,
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

        logger.info("P300 test complete")

    def test_ssvep_offline(self):
        # Get the SSVEP example data from ./examples/data
        ssvep_xdf_path = os.path.join(data_folder_path, "ssvep_example.xdf")

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
        data = EEG_data(classifier)
        data.load_offline_eeg_data(filename=ssvep_xdf_path, print_output=True)

        # Run main loop, this will do all of the classification for online or offline
        data.main(
            online=False,
            training=True,
            pp_type="bandpass",
            pp_low=3,
            pp_high=50,
            pp_order=5,
            print_markers=True,
            print_training=True,
            print_fit=True,
            print_performance=True,
            print_predict=True,
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

        logger.info("SSVEP test complete")


if __name__ == "__main__":
    unittest.main()

logger.debug("Finished test_logger.py")