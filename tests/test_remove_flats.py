import unittest
import os

from bci_essentials.io.xdf_sources import XdfMarkerSource, XdfEegSource
from bci_essentials.bci_controller import BciController
from bci_essentials.paradigm.mi_paradigm import MiParadigm
from bci_essentials.paradigm.p300_paradigm import P300Paradigm
from bci_essentials.paradigm.ssvep_paradigm import SsvepParadigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.mi_classifier import MiClassifier
from bci_essentials.classification.erp_rg_classifier import ErpRgClassifier
from bci_essentials.classification.ssvep_riemannian_mdm_classifier import (
    SsvepRiemannianMdmClassifier,
)
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger(name="test_smoke")


class TestRemoveFlats(unittest.TestCase):
    def test_mi_remove_flats(self):
        # Get the DSI24 MI example data from ./examples/data
        # This data contains a zero channel "TRG", which much be removed
        xdf_path = os.path.join(os.path.dirname(__file__), "data", "mi_smoke_dsi24.xdf")
        eeg_source = XdfEegSource(xdf_path)
        marker_source = XdfMarkerSource(xdf_path)

        paradigm = MiParadigm(live_update=True, iterative_training=False)
        data_tank = DataTank()

        # Select a classifier
        classifier = MiClassifier()
        classifier.set_mi_classifier_settings(
            n_splits=5,
            type="TS",
            random_seed=35,
            remove_flats=True,
            covariance_estimator="scm",
        )

        # Load the data
        data = BciController(classifier, eeg_source, marker_source, paradigm, data_tank)

        # Run main loop, this will do all of the classification for online or offline
        data.setup(
            online=False,
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

    def test_p300_remove_flats(self):
        # Get the DSI24 P300 example data from ./examples/data
        # This data contains a zero channel "TRG", which much be removed
        xdf_path = os.path.join(
            os.path.dirname(__file__), "data", "p300_smoke_dsi24.xdf"
        )
        eeg_source = XdfEegSource(xdf_path)
        marker_source = XdfMarkerSource(xdf_path)

        paradigm = P300Paradigm()
        data_tank = DataTank()

        # Select a classifier
        classifier = ErpRgClassifier()
        classifier.set_p300_clf_settings(
            n_splits=5,
            lico_expansion_factor=4,
            oversample_ratio=0,
            undersample_ratio=0,
            random_seed=35,
            remove_flats=True,
            covariance_estimator="scm",
        )

        # Load the data
        data = BciController(classifier, eeg_source, marker_source, paradigm, data_tank)

        # Run main loop, this will do all of the classification for online or offline
        data.setup(
            online=False,
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


if __name__ == "__main__":
    unittest.main()
