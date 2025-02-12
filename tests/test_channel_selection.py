import unittest
import os
import numpy as np

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
from bci_essentials.channel_selection import channel_selection_by_method


import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM


# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger(name="test_channel_selection")

# Tests channel selection using 10 channels, with growing amounts of noise
# First create an ideal signal, a sinusoid with frequency 8 Hz and amplitude 1
# Then add noise to the signal, with increasing standard deviation
# Finally, test the channel selection algorithm on the noisy signal

# Create the ideal signal
fs = 128
t = np.arange(0, 1, 1 / fs)

import matplotlib.pyplot as plt
# Create epochs of 

X = np.zeros((10, 10, len(t)))
y = np.zeros(10)

# Create a spline
trend = [(0.01 * (-0.08*s +2.0)**3.0) + (0.05 * s) + 1.0 for s in range(1,129)]




channel_labels = ["ch" + str(i) for i in range(1,11)]

for i in range(10):
    for j in range(10):
        if i % 2 == 0:
            y[i] = 0

            ideal_signal = np.sin(2 * np.pi * 8 * t)
            signal = ideal_signal + np.random.normal(0, 2, len(t))
            for k in range(j):
                signal += np.random.normal(0, 2, len(t))

            X[i, j, :] = signal

        else:
            y[i] = 1

            ideal_signal = np.sin(2 * np.pi * 8 * t) * trend
            signal = ideal_signal + np.random.normal(0, 2, len(t))
            for k in range(j):
                signal += np.random.normal(0, 2, len(t))

            X[i, j, :] = signal

# # Plot the first 10 epochs
# for i in range(10):
#     plt.plot(t, X[1, i, :] + i*20)

# plt.legend(channel_labels)
# plt.show()


def _test_kernel(subX, suby):
        """Test kernel for channel selection.

        Parameters
        ----------
        subX : numpy.ndarray
            EEG data for training.
            3D array with shape = (`n_epochs`, `n_channels`, `n_samples`).
        suby : numpy.ndarray
            Labels for training data.
            1D array with shape = (`n_epochs`, ).

        Returns
        -------
        model : classifier
            The trained classification model.
        preds : numpy.ndarray
            The predictions from the model.
            1D array with the same shape as `suby`.
        accuracy : float
            The accuracy of the trained classification model.
        precision : float
            The precision of the trained classification model.
        recall : float
            The recall of the trained classification model.

        """
        n_splits = 5
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        )

        mdm = MDM(metric=dict(mean="riemann", distance="riemann"))
        clf_model = Pipeline([("MDM", mdm)])

        preds = np.zeros_like(suby)


        for train_idx, test_idx in cv.split(subX, suby):
            clf = clf_model

            X_train, X_test = subX[train_idx], subX[test_idx]
            # y_test not implemented
            y_train = suby[train_idx]

            # get the covariance matrices for the training set
            X_train_cov = Covariances().transform(X_train)
            X_test_cov = Covariances().transform(X_test)

            # fit the classsifier
            clf.fit(X_train_cov, y_train)
            preds[test_idx] = clf.predict(X_test_cov)

        accuracy = sum(preds == suby) / len(preds)
        precision = precision_score(suby, preds, average="micro")
        recall = recall_score(suby, preds, average="micro")

        model = clf

        return model, preds, accuracy, precision, recall




class TestChannelSelection(unittest.TestCase):
    def test_sfs(self):
        # Test SFS
        selected_channels = channel_selection_by_method(
            kernel_func = _test_kernel, 
            X = X, 
            y = y, 
            channel_labels = channel_labels,
            method="SFS", 
            initial_channels=[],
            max_time=999,
            min_channels=0,
            max_channels=20, 
            performance_delta=-1,
            n_jobs=-1, 
            record_performance=True
        )
        
        print(selected_channels)

        self.assertEqual(len(selected_channels), 5)




#         # Get the MI example data from ./examples/data
#         xdf_path = os.path.join(os.path.dirname(__file__), "data", "mi_smoke.xdf")
#         eeg_source = XdfEegSource(xdf_path)
#         marker_source = XdfMarkerSource(xdf_path)

#         paradigm = MiParadigm(live_update=True, iterative_training=False)
#         data_tank = DataTank()

#         # Select a classifier
#         classifier = MiClassifier()
#         classifier.set_mi_classifier_settings(
#             n_splits=5, type="TS", random_seed=35, channel_selection="riemann"
#         )

#         # Load the data
#         data = BciController(classifier, eeg_source, marker_source, paradigm, data_tank)

#         # Run main loop, this will do all of the classification for online or offline
#         data.setup(
#             online=False,
#         )
#         data.run()

#         # Check that a model was trained
#         self.assertIsNotNone(classifier.clf)

#         # Check that accuracy, precision, recall and covariance matrix exist
#         self.assertIsNotNone(classifier.offline_accuracy)
#         self.assertIsNotNone(classifier.offline_precision)
#         self.assertIsNotNone(classifier.offline_recall)
#         self.assertIsNotNone(classifier.offline_cm)

#         # Check that the list of class predictions exists
#         self.assertIsNotNone(classifier.predictions)

#         logger.info("MI test complete")

#     def test_p300_offline(self):
#         # Get the P300 example data from ./examples/data
#         xdf_path = os.path.join(os.path.dirname(__file__), "data", "p300_smoke.xdf")
#         eeg_source = XdfEegSource(xdf_path)
#         marker_source = XdfMarkerSource(xdf_path)

#         paradigm = P300Paradigm()
#         data_tank = DataTank()

#         # Select a classifier
#         classifier = ErpRgClassifier()
#         classifier.set_p300_clf_settings(
#             n_splits=5,
#             lico_expansion_factor=4,
#             oversample_ratio=0,
#             undersample_ratio=0,
#             random_seed=35,
#         )

#         # Load the data
#         data = BciController(classifier, eeg_source, marker_source, paradigm, data_tank)

#         # Run main loop, this will do all of the classification for online or offline
#         data.setup(
#             online=False,
#         )
#         data.run()

#         # Check that a model was trained
#         self.assertIsNotNone(classifier.clf)

#         # Check that accuracy, precision, recall and covariance matrix exist
#         self.assertIsNotNone(classifier.offline_accuracy)
#         self.assertIsNotNone(classifier.offline_precision)
#         self.assertIsNotNone(classifier.offline_recall)
#         self.assertIsNotNone(classifier.offline_cm)

#         # Check that the list of class predictions exists
#         self.assertIsNotNone(classifier.predictions)

#         logger.info("P300 test complete")

#     def test_ssvep_offline(self):
#         # Get the SSVEP example data from ./examples/data
#         xdf_path = os.path.join(os.path.dirname(__file__), "data", "ssvep_smoke.xdf")
#         eeg_source = XdfEegSource(xdf_path)
#         marker_source = XdfMarkerSource(xdf_path)

#         paradigm = SsvepParadigm()
#         data_tank = DataTank()

#         # Select a classifier
#         classifier = SsvepRiemannianMdmClassifier()
#         classifier.set_ssvep_settings(
#             n_splits=3,
#             random_seed=35,
#             n_harmonics=2,
#             f_width=0.2,
#             covariance_estimator="scm",
#         )
#         classifier.target_freqs = [
#             24,
#             20.57143,
#             18,
#             16,
#             14.4,
#             12,
#             10.28571,
#             9,
#             8,
#             7.2,
#             6,
#             4.965517,
#         ]

#         # Load the data
#         data = BciController(classifier, eeg_source, marker_source, paradigm, data_tank)

#         # Run main loop, this will do all of the classification for online or offline
#         data.setup(
#             online=False,
#         )
#         data.run()

#         # Check that a model was trained
#         self.assertIsNotNone(classifier.clf)

#         # Check that accuracy, precision, recall and covariance matrix exist
#         self.assertIsNotNone(classifier.offline_accuracy)
#         self.assertIsNotNone(classifier.offline_precision)
#         self.assertIsNotNone(classifier.offline_recall)
#         self.assertIsNotNone(classifier.offline_cm)

#         # Check that the list of class predictions exists
#         self.assertIsNotNone(classifier.predictions)

#         logger.info("SSVEP test complete")


if __name__ == "__main__":
    unittest.main()
