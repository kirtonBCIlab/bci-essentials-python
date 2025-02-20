import unittest
import numpy as np
import time

from bci_essentials.utils.logger import Logger  # Logger wrapper
from bci_essentials.channel_selection import channel_selection_by_method

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM


# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger(name="test_channel_selection")

# Tests channel selection using 10 channels, with growing amounts of noise.
# The "signal" is a spoline that varies the power of a sine wave.

# Create the ideal signal
fs = 128
t = np.arange(0, 1, 1 / fs)
X = np.zeros((1000, 10, len(t)))
y = np.zeros(1000)

# Create a spline
trend = [
    ((0.01 * (-0.08 * s + 2.0) ** 3.0) + (0.05 * s) + 1.0) ** 0.2 for s in range(1, 129)
]

channel_labels = ["ch" + str(i) for i in range(1, 11)]

for i in range(1000):  # 1000 epochs
    for j in range(10):  # 10 channels
        if i % 2 == 0:
            y[i] = 0

            ideal_signal = np.sin(2 * np.pi * 8 * t)
            signal = ideal_signal + np.random.normal(0, 2, len(t))
            for k in range(j):
                signal += np.random.normal(0, 2, len(t))

            X[i, j, :] = signal

        else:
            y[i] = 1

            ideal_signal = [
                np.sin(2 * np.pi * 8 * t) * (trend[t] ** 1 / (j + 1))
                for t in range(len(t))
            ]  # Trend size is smaller in higher channels
            signal = ideal_signal + np.random.normal(0, 2, len(t))
            for k in range(j):
                signal += np.random.normal(
                    0, 2, len(t)
                )  # More noise added in higher channels

            X[i, j, :] = signal


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
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

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
    def test_time_limit(self):
        # Test params for time
        min_channels = 0
        max_channels = 10
        max_time = 5

        # Test SFS for time
        time_start = time.time()
        selection_output = channel_selection_by_method(
            kernel_func=_test_kernel,
            X=X,
            y=y,
            channel_labels=channel_labels,
            method="SFS",
            initial_channels=["ch1", "ch2"],
            max_time=max_time,
            min_channels=min_channels,
            max_channels=max_channels,
            performance_delta=-1,
            n_jobs=-1,
            record_performance=True,
        )
        time_end = time.time()

        best_subset = selection_output[0]

        # Check that the best subset is within the correct range
        self.assertGreaterEqual(len(best_subset), min_channels)
        self.assertLessEqual(len(best_subset), max_channels)

        # Check time taken
        allowable_buffer = 1.0
        self.assertLessEqual(time_end - time_start, max_time + allowable_buffer)

    def test_max_channels(self):
        # Test params for max channels
        min_channels = 1
        max_channels = 6
        max_time = 100

        # Test SFS for max_channels
        selection_output = channel_selection_by_method(
            kernel_func=_test_kernel,
            X=X,
            y=y,
            channel_labels=channel_labels,
            method="SFS",
            initial_channels=[],
            max_time=max_time,
            min_channels=min_channels,
            max_channels=max_channels,
            performance_delta=-1,
            n_jobs=-1,
            record_performance=True,
        )

        best_subset = selection_output[0]
        results_df = selection_output[6]

        # Check that the best subset is within the correct range
        self.assertGreaterEqual(len(best_subset), min_channels)
        self.assertLessEqual(len(best_subset), max_channels)

        # Check that the algorithm didn't check combinations outside of the range
        min_channels_tried = int(results_df["N Channels"].min())
        max_channels_tried = int(results_df["N Channels"].max())
        self.assertGreaterEqual(min_channels_tried, min_channels)
        self.assertLessEqual(max_channels_tried, max_channels)

        # Test SFFS for max_channels
        selection_output = channel_selection_by_method(
            kernel_func=_test_kernel,
            X=X,
            y=y,
            channel_labels=channel_labels,
            method="SFS",
            initial_channels=["ch1", "ch2", "ch3", "ch4"],
            max_time=max_time,
            min_channels=min_channels,
            max_channels=max_channels,
            performance_delta=-1,
            n_jobs=-1,
            record_performance=True,
        )

        best_subset = selection_output[0]
        results_df = selection_output[6]

        # Check that the best subset is within the correct range
        self.assertGreaterEqual(len(best_subset), min_channels)
        self.assertLessEqual(len(best_subset), max_channels)

        # Check that the algorithm didn't check combinations outside of the range
        min_channels_tried = int(results_df["N Channels"].min())
        max_channels_tried = int(results_df["N Channels"].max())
        self.assertGreaterEqual(min_channels_tried, min_channels)
        self.assertLessEqual(max_channels_tried, max_channels)

    def test_min_channels(self):
        # Test params for max channels
        min_channels = 2
        max_channels = 10
        max_time = 100

        # Test SBS for min_channels
        selection_output = channel_selection_by_method(
            kernel_func=_test_kernel,
            X=X,
            y=y,
            channel_labels=channel_labels,
            method="SBS",
            initial_channels=[],
            max_time=max_time,
            min_channels=min_channels,
            max_channels=max_channels,
            performance_delta=-1,
            n_jobs=-1,
            record_performance=True,
        )

        best_subset = selection_output[0]
        results_df = selection_output[6]

        # Check that the best subset is within the correct range
        self.assertGreaterEqual(len(best_subset), min_channels)
        self.assertLessEqual(len(best_subset), max_channels)

        # Check that the algorithm didn't check combinations outside of the range
        min_channels_tried = int(results_df["N Channels"].min())
        max_channels_tried = int(results_df["N Channels"].max())
        self.assertGreaterEqual(min_channels_tried, min_channels)
        self.assertLessEqual(max_channels_tried, max_channels)

        # Test SBFS for min_channels
        selection_output = channel_selection_by_method(
            kernel_func=_test_kernel,
            X=X,
            y=y,
            channel_labels=channel_labels,
            method="SBFS",
            initial_channels=["ch1", "ch2", "ch3", "ch4"],
            max_time=max_time,
            min_channels=min_channels,
            max_channels=max_channels,
            performance_delta=-1,
            n_jobs=-1,
            record_performance=True,
        )

        best_subset = selection_output[0]
        results_df = selection_output[6]

        # Check that the best subset is within the correct range
        self.assertGreaterEqual(len(best_subset), min_channels)
        self.assertLessEqual(len(best_subset), max_channels)

        # Check that the algorithm didn't check combinations outside of the range
        min_channels_tried = int(results_df["N Channels"].min())
        max_channels_tried = int(results_df["N Channels"].max())
        self.assertGreaterEqual(min_channels_tried, min_channels)
        self.assertLessEqual(max_channels_tried, max_channels)

        best_subset = selection_output[0]
        results_df = selection_output[6]

    # Test performance delta
    def test_performance_delta(self):
        # Test params for time
        min_channels = 0
        max_channels = 10
        max_time = 100
        performance_delta = 0.01

        # Test SFS for time
        selection_output = channel_selection_by_method(
            kernel_func=_test_kernel,
            X=X,
            y=y,
            channel_labels=channel_labels,
            method="SFS",
            initial_channels=[],
            max_time=max_time,
            min_channels=min_channels,
            max_channels=max_channels,
            performance_delta=performance_delta,
            n_jobs=-1,
            record_performance=True,
        )

        results_df = selection_output[6]

        # Check that the algorithm didn't check combinations outside of the range
        scores = results_df["Accuracy"]
        score_difs = np.diff(scores)
        self.assertGreaterEqual(
            score_difs[:-1].max(), performance_delta
        )  # Check that all but the last score difs are greater than performance delta


if __name__ == "__main__":
    unittest.main()
