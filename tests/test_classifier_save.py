import unittest
import os
import numpy as np

from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.bci_controller import BciController
from bci_essentials.paradigm.mi_paradigm import MiParadigm
from bci_essentials.paradigm.p300_paradigm import P300Paradigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.mi_classifier import MiClassifier
from bci_essentials.classification.erp_rg_classifier import ErpRgClassifier
from bci_essentials.session_saving import save_classifier, load_classifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger(name="test_classifier_save")


class TestClassifierSave(unittest.TestCase):
    def test_mi_manual_classifier_save(self):
        xdf_path = os.path.join(os.path.dirname(__file__), "data", "mi_smoke.xdf")
        eeg_source1 = XdfEegSource(xdf_path)
        marker_source1 = XdfMarkerSource(xdf_path)

        paradigm = MiParadigm(live_update=True, iterative_training=False)
        data_tank1 = DataTank()

        # Select a classifier
        classifier1 = MiClassifier()
        classifier1.set_mi_classifier_settings(
            n_splits=5,
            type="TS",
            random_seed=35,
            channel_selection="riemann",
            covariance_estimator="oas",
        )

        # Load the data
        data1 = BciController(
            classifier1, eeg_source1, marker_source1, paradigm, data_tank1
        )

        # Run main loop, this will do all of the classification for online or offline
        data1.setup(
            online=False,
        )
        data1.run()

        # Save the classifier
        save_classifier(classifier1, "test_mi_classifier.pkl")

        #
        data_tank2 = DataTank()

        # Load the classifier
        classifier2 = load_classifier("test_mi_classifier.pkl")

        # Create a new BciController object, recreate xdf sources to reload files
        eeg_source2 = XdfEegSource(xdf_path)
        marker_source2 = XdfMarkerSource(xdf_path)
        data2 = BciController(
            classifier2, eeg_source2, marker_source2, paradigm, data_tank2
        )

        # Check that all values of the classifier's numpy arrays are the same
        self.assertTrue(np.array_equal(classifier1.X, classifier2.X))

        logger.info("Classifier reloaded")

        # Delete classifier predictions
        classifier2.predictions = []

        # Run the main loop
        data2.setup(
            online=False,
            train_complete=True,
            train_lock=True,
        )
        data2.run()

        # Check that X and y are the same
        self.assertTrue(np.array_equal(classifier1.X, classifier2.X))
        self.assertTrue(np.array_equal(classifier1.y, classifier2.y))

        # Check that the last predictions are the same (the last 78 predictions should be the same)
        self.assertTrue(
            np.array_equal(
                classifier1.predictions[-78:],
                classifier2.predictions[-78:],
            )
        )

        logger.info("MI classifier save/load test passed")

    def test_p300_manual_classifier_save(self):
        xdf_path = os.path.join(os.path.dirname(__file__), "data", "p300_smoke.xdf")
        eeg_source1 = XdfEegSource(xdf_path)
        marker_source1 = XdfMarkerSource(xdf_path)

        paradigm = P300Paradigm()
        data_tank1 = DataTank()

        # Select a classifier
        classifier1 = ErpRgClassifier()
        classifier1.set_p300_clf_settings(
            n_splits=5,
            lico_expansion_factor=4,
            oversample_ratio=0,
            undersample_ratio=0,
            random_seed=35,
        )

        # Load the data
        data1 = BciController(
            classifier1, eeg_source1, marker_source1, paradigm, data_tank1
        )

        # Run main loop, this will do all of the classification for online or offline
        data1.setup(
            online=False,
        )
        data1.run()

        # Save the classifier model
        save_classifier(classifier1, "test_p300_classifier.pkl")

        data_tank2 = DataTank()

        # Load the classifier
        classifier2 = load_classifier("test_p300_classifier.pkl")

        # Create a new ErpData object, recreate xdf sources to reload files
        eeg_source2 = XdfEegSource(xdf_path)
        marker_source2 = XdfMarkerSource(xdf_path)
        data2 = BciController(
            classifier2, eeg_source2, marker_source2, paradigm, data_tank2
        )

        # Check that all values of the classifier's numpy arrays are the same
        self.assertTrue(np.array_equal(classifier1.X, classifier2.X))

        logger.info("Classifier reloaded")

        # Delete classifier predictions
        classifier2.predictions = []

        # Run the main loop
        data2.setup(
            online=False,
            train_complete=True,
            train_lock=True,
        )
        data2.run()

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
