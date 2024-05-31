import unittest
import os
import numpy as np

from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.bci_controller import BciController
from bci_essentials.paradigm.mi_paradigm import MiParadigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.null_classifier import NullClassifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger(name="test_classifier_save")


class TestRawTrialSave(unittest.TestCase):
    def test_xy_trial_save(self):
        xdf_path = os.path.join(os.path.dirname(__file__), "data", "mi_smoke.xdf")
        eeg_source1 = XdfEegSource(xdf_path)
        marker_source1 = XdfMarkerSource(xdf_path)

        paradigm1 = MiParadigm(live_update=True, iterative_training=False)
        data_tank1 = DataTank()

        # Select a classifier
        classifier1 = NullClassifier()

        # Load the data
        data1 = BciController(
            classifier1, eeg_source1, marker_source1, paradigm1, data_tank1
        )

        # Run main loop, this will do all of the classification for online or offline
        data1.setup(
            online=False,
        )
        data1.run()

        # Create a temp .npy file to save the raw EEG trials and labels
        data_tank1.save_epochs_as_npz("raw_trial_eeg.npz")

        # Load the Raw EEG Trials and labels from the .npy file
        loaded_npz = np.load("raw_trial_eeg.npz")

        X = loaded_npz["X"]
        y = loaded_npz["y"]

        # Delete the temp .npz file
        loaded_npz.close()
        os.remove("raw_trial_eeg.npz")

        # Check that all values of the classifier's numpy arrays are the same
        self.assertTrue(np.array_equal(data_tank1.epochs, X))
        self.assertTrue(np.array_equal(data_tank1.labels, y))


if __name__ == "__main__":
    unittest.main()
