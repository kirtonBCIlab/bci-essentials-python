import unittest
import os
import numpy as np

from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.eeg_data import EegData
from bci_essentials.classification.null_classifier import NullClassifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger(name="test_classifier_save")


class TestRawTrialSave(unittest.TestCase):
    def test_xy_trial_save(self):
        xdf_path = os.path.join(os.path.dirname(__file__), "data", "mi_smoke.xdf")
        eeg_source1 = XdfEegSource(xdf_path)
        marker_source1 = XdfMarkerSource(xdf_path)

        # Select a classifier
        classifier1 = NullClassifier()

        # Load the data
        data1 = EegData(classifier1, eeg_source1, marker_source1)

        # Run main loop, this will do all of the classification for online or offline
        data1.setup(
            online=False,
            training=True,
            pp_low=5,
            pp_high=50,
            pp_order=5,
        )
        data1.run()

        # Create a temp .npy file to save the raw EEG trials and labels
        data1.save_trials_as_npz("raw_trial_eeg.npz")

        # Load the Raw EEG Trials and labels from the .npy file
        loaded_npz = np.load("raw_trial_eeg.npz")

        X = loaded_npz["X"]
        y = loaded_npz["y"]

        # Delete the temp .npz file
        loaded_npz.close()
        os.remove("raw_trial_eeg.npz")

        # Check that all values of the classifier's numpy arrays are the same
        self.assertTrue(np.array_equal(data1.raw_eeg_trials, X))
        self.assertTrue(np.array_equal(data1.labels, y))


if __name__ == "__main__":
    unittest.main()
