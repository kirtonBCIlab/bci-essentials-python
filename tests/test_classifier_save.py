import unittest
import os
import numpy as np

from bci_essentials.bci_data import EEG_data
from bci_essentials.classification.mi_classifier import MI_classifier
from bci_essentials.session_saving import save_classifier, load_classifier

data_folder_path = os.path.join("examples", "data")

class TestClassifierSave(unittest.TestCase):
    def test_manual_classifier_save(self):
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
            n_splits=5, type="TS", random_seed=35, channel_selection="riemann", covariance_estimator="oas"
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
        save_classifier(mi_data1.classifier, 'test_mi_classifier.pkl')

        # Create a new EEG_data object
        mi_data2 = EEG_data()

        # Load the data
        mi_data2.load_offline_eeg_data(filename=mi_xdf_path, print_output=False)

        # Select a classifier
        mi_data2.classifier = load_classifier('test_mi_classifier.pkl')

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
        self.assertTrue(np.array_equal(mi_data1.classifier.predictions[-78:], mi_data2.classifier.predictions[-78:]))

        print("MI classifier save/load test passed")


if __name__ == "__main__":
    unittest.main()