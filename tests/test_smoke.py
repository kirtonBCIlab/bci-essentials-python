import unittest
import os

from bci_essentials.io.xdf_sources import XdfMarkerSource, XdfEegSource
from bci_essentials.eeg_data import EegData
from bci_essentials.erp_data import ErpData
from bci_essentials.classification.mi_classifier import MiClassifier
from bci_essentials.classification.erp_rg_classifier import ErpRgClassifier
from bci_essentials.classification.ssvep_riemannian_mdm_classifier import (
    SsvepRiemannianMdmClassifier,
)
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger(name="test_smoke")


class TestSmoke(unittest.TestCase):
    def test_mi_offline(self):
        # Get the MI example data from ./examples/data
        xdf_path = os.path.join(os.path.dirname(__file__), "data", "mi_smoke.xdf")
        eeg_source = XdfEegSource(xdf_path)
        marker_source = XdfMarkerSource(xdf_path)

        # Select a classifier
        classifier = MiClassifier()
        classifier.set_mi_classifier_settings(
            n_splits=5, type="TS", random_seed=35, channel_selection="riemann"
        )

        # Load the data
        data = EegData(classifier, eeg_source, marker_source)

        # Run main loop, this will do all of the classification for online or offline
        data.setup(
            online=False,
            training=True,
            pp_low=5,
            pp_high=50,
            pp_order=5,
        )
        data.run()

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
        xdf_path = os.path.join(os.path.dirname(__file__), "data", "p300_smoke.xdf")
        eeg_source = XdfEegSource(xdf_path)
        marker_source = XdfMarkerSource(xdf_path)

        # Select a classifier
        classifier = ErpRgClassifier()
        classifier.set_p300_clf_settings(
            n_splits=5,
            lico_expansion_factor=4,
            oversample_ratio=0,
            undersample_ratio=0,
            random_seed=35,
        )

        # Load the data
        data = ErpData(classifier, eeg_source, marker_source)

        # Run main loop, this will do all of the classification for online or offline
        data.setup(
            online=False,
            training=True,
            pp_low=0.1,
            pp_high=10,
            pp_order=5,
            plot_erp=False,
            trial_start=0.0,
            trial_end=0.8,
            max_num_options=9,
            max_trials_per_option=16,
            max_decisions=20,
        )
        data.run()

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
        xdf_path = os.path.join(os.path.dirname(__file__), "data", "ssvep_smoke.xdf")
        eeg_source = XdfEegSource(xdf_path)
        marker_source = XdfMarkerSource(xdf_path)

        # Select a classifier
        classifier = SsvepRiemannianMdmClassifier()
        classifier.set_ssvep_settings(
            n_splits=3,
            random_seed=35,
            n_harmonics=2,
            f_width=0.2,
            covariance_estimator="scm",
        )

        # Load the data
        data = EegData(classifier, eeg_source, marker_source)

        # Run main loop, this will do all of the classification for online or offline
        data.setup(
            online=False,
            training=True,
            pp_type="bandpass",
            pp_low=3,
            pp_high=50,
            pp_order=5,
        )
        data.run()

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
