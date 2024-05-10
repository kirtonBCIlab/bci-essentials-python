"""**Generic classifier class for BCI Essentials**

Used as Parent classifier class for other classifiers.

"""

# Stock libraries
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


@dataclass
class Prediction:
    """Prediction data returned by GenericClassifer.predict()

    labels : list
        List of the predicted class labels.
        - Default is `[]`.

    probabilities : list
        List of probabilities for each class label. If the classifier can't
        provide probabilities, this will be an empty list `[]`.
        - Default is `[]`

    """

    labels: list = field(default_factory=list)
    probabilities: list = field(default_factory=list)


class GenericClassifier(ABC):
    """The base generic classifier class for other classifiers."""

    def __init__(self, training_selection=0, subset=[]):
        """Initializes `GenericClassifier` class.

        Parameters
        ----------
        training_selection : int, *optional*
            Integer representing the object selected for training.
            - Default is `0`.
        subset : list of `int` or `str`, *optional*
            List of indices (int) or labels (str) of the desired channels.
            - Default is `[]`.

        Attributes
        ----------
        X : numpy.ndarray
            Input features (training data).
            3D numpy array with shape = (`n_samples`, `n_channels`, `n_trials`).
            - Initial value is `np.ndarray([0])`.
        y : numpy.ndarray
            Target labels corresponding to input features in `X`.
            1D numpy array with shape = (`n_samples`, ).
            - Initial value is `np.ndarray([0])`.
        subset_defined : bool
            Flag indicating whether a subset is defined.
            - Initial value is `False`.
        subset : list of `int` or `str`
            List of indices (int) or labels (str) of the desired channels.
            - Initial value is parameter `subset`.
        channel_labels : list of `str`
            Channel labels from the entire EEG montage.
            - Initial value is `[]`.
        channel_selection_setup : bool
            FLag indicating whether channel selection is set up.
            - Initial value is `False`.
        offline_accuracy : list of `float`
            Stores offline accuracy values during training.
            - Initial value is `[]`.
        offline_precision : list of `float`
            Stores offline precision values during training.
            - Initial value is `[]`.
        offline_recall : list of `float`
            Stores offline recall values during training.
            - Initial value is `[]`.
        offline_trial_count : int
            Counter to keep track of the number of offline trials
            - Initial value is `0`.
        offline_trial_counts : list of `int`
            List to store the counts of offline trials.
            i.e. `offline_trial_count' values.
            - Initial value is `[]`.
        next_fit_trial : int
            Counter to track the next trial for fitting.
            - Initial value is `0`.
        predictions : list
            Stores predications made during training or testing
            - Initial value is `[]`.
        pred_probas : list of `float`
            List to store predication probabilities during testing.
            - Initial value is `[]`.

        """
        logger.info("Initializing the classifier")
        self.X = np.ndarray([0])
        """@private (This is just for the API docs, to avoid double listing."""
        self.y = np.ndarray([0])
        """@private (This is just for the API docs, to avoid double listing."""

        self.subset_defined = False
        """@private (This is just for the API docs, to avoid double listing."""
        self.subset = subset
        """@private (This is just for the API docs, to avoid double listing."""
        self.channel_labels = []
        """@private (This is just for the API docs, to avoid double listing."""
        self.channel_selection_setup = False
        """@private (This is just for the API docs, to avoid double listing."""

        # Lists for plotting classifier performance over time
        self.offline_accuracy = []
        """@private (This is just for the API docs, to avoid double listing."""
        self.offline_precision = []
        """@private (This is just for the API docs, to avoid double listing."""
        self.offline_recall = []
        """@private (This is just for the API docs, to avoid double listing."""
        self.offline_trial_count = 0
        """@private (This is just for the API docs, to avoid double listing."""
        self.offline_trial_counts = []
        """@private (This is just for the API docs, to avoid double listing."""

        # For iterative fitting,
        self.next_fit_trial = 0
        """@private (This is just for the API docs, to avoid double listing."""

        # Keep track of predictions
        self.predictions = []
        """@private (This is just for the API docs, to avoid double listing."""
        self.pred_probas = []
        """@private (This is just for the API docs, to avoid double listing."""

    def get_subset(self, X=[], subset=[], channel_labels=[]):
        """Get a subset of X according to labels or indices.

        Parameters
        ----------
        X : numpy.ndarray, *optional*
            3D array containing data with `float` type.

            shape = (`n_trials`,`n_channels`,`n_samples`)
            - Default is `[]`.
        subset : list of `int` or `str`, *optional*
            List of indices (int) or labels (str) of the desired channels.
            - Default is `[]`.
        channel_labels : list of `str`, *optional*
            Channel labels from the entire EEG montage.
            - Default is `[]`.

        Returns
        -------
        X : numpy.ndarray
            Subset of input `X` according to labels or indices.
            3D array containing data with `float` type.

            shape = (`n_trials`,`n_channels`,`n_samples`)

        """

        # Check for self.subset and/or self.channel_labels

        # Init
        subset_indices = []

        # Copy the indices based on subset
        try:
            # Check if we can use subset indices
            if self.subset == []:
                return X

            if type(self.subset[0]) is int:
                logger.info("Using subset indices")

                subset_indices = self.subset

            # Or channel labels
            if type(self.subset[0]) is str:
                logger.info("Using channel labels and subset labels")

                # Replace indices with those described by labels
                for sl in self.subset:
                    subset_indices.append(self.channel_labels.index(sl))

            # Return for the given indices
            try:
                if sum(X.shape) == 0:
                    new_X = self.X[:, subset_indices, :]
                    self.X = new_X
                else:
                    new_X = X[:, subset_indices, :]
                    X = new_X
                    return X

            except Exception:
                if sum(X.shape) == 0:
                    new_X = self.X[subset_indices, :]
                    self.X = new_X

                else:
                    new_X = X[subset_indices, :]
                    X = new_X
                    return X

        # notify if failed
        except Exception:
            logger.warning("something went wrong, no subset taken")
            return X

    def setup_channel_selection(
        self,
        method="SBS",
        metric="accuracy",
        iterative_selection=False,
        initial_channels=[],  # wrapper setup
        max_time=999,
        min_channels=1,
        max_channels=999,
        performance_delta=0.001,  # stopping criterion
        n_jobs=1,
        record_performance=False,
    ):
        """Setup channel selection parameters.

        Parameters
        ----------
        method : str, *optional*
            The method used to add or remove channels.
            - Default is `"SBS"`.
        metric : str, *optional*
            The metric used to measure performance.
            - Default is `"accuracy"`.
        iterative_selection : bool, *optional*
            Whether or not to use the previously selected subset for the initial subset.
            Default is `False`.
        initial_channels : type, *optional*
            List of channels to use as initial subset for selection.
            If empty, `initial_channels` is set to all available channels.
            - Default is `[]`.
        max_time : int, *optional*
            Maximum time in seconds allowed for channel selection.
            - Default is `999`.
        min_channels : int, *optional*
            Minimum number of channels to select during channel selection.
            - Default is `1`.
        max_channels : int, *optional*
            Maximum number of channels allowed in the final subset.
            - Default is `999`.
        performance_delta : float, *optional*
            Smallest performance increment to allow continue of the search.
            - Default is `0.001`.
        n_jobs : int, *optional*
            The number of threads to dedicate to this calculation.
            - Default is `1`.
        record_performance : bool, *optional*
            Decides whether or not to record performance of channel selection.
            - Default is `False`.

        Returns
        -------
        `None`

        """
        # Add these to settings later
        if initial_channels == []:
            self.chs_initial_subset = self.channel_labels
        else:
            self.chs_initial_subset = initial_channels
        self.chs_method = method  # method to add/remove channels
        self.chs_metric = metric  # metric by which to measure performance
        self.chs_iterative_selection = iterative_selection  # whether or not to use the previously selected subset for the initial subset
        self.chs_n_jobs = n_jobs  # number of threads
        self.chs_max_time = max_time  # max time in seconds
        self.chs_min_channels = min_channels  # minimum number of channels
        self.chs_max_channels = max_channels  # maximum number of channels
        self.chs_performance_delta = performance_delta  # smallest performance increment to justify continuing search
        self.chs_record_performance = record_performance  # record performance

        self.channel_selection_setup = True

    # add training data, to the training set using a decision block and a label
    def add_to_train(self, decision_block, labels, num_options=0, meta=[]):
        """Add training data to the training set using a decision block
        and a label.

        Parameters
        ----------
        decision_block : numpy.ndarray
            Decision block containing EEG data for training.
            3D array with shape = (`n_epochs`, `n_channels`, `n_samples`).
        labels : numpy.ndarray
            Labels corresponding to each epoch in `decision_block`.
            1D array with shape = (`n_epochs`, ).
        num_options : int, *optional*
            Number of options available for each trial.
            - Default is `0`.
        meta : list, *optional*
            Additional metadata related to the training data.
            - Default is `[]`.

        Returns
        -------
        `None`

        """
        logger.debug("Adding to training set")
        # n = number of channels
        # m = number of samples
        # p = number of epochs
        p, n, m = decision_block.shape

        self.num_options = num_options
        self.meta = meta

        if self.X.size == 0:
            self.X = decision_block
            self.y = labels

        else:
            self.X = np.append(self.X, decision_block, axis=0)
            self.y = np.append(self.y, labels, axis=0)

    # # predict a label based on a decision block
    # # This doesn't seem to be used anywhere
    # def predict_decision_block(self, decision_block) -> Prediction:
    #     """Predict a label based on a decision block.

    #     Parameters
    #     ----------
    #     decision_block : numpy.ndarray
    #         Decision block containing EEG data for training.
    #         3D array with shape = (`n_epochs`, `n_channels`, `n_samples`).

    #     Returns
    #     -------
    #     prediction : Prediction
    #         A Predication object containing the predicated label and
    #         prediction probabilities.

    #     """
    #     decision_block_subset = self.get_subset(
    #         decision_block, self.subset, self.channel_labels
    #     )

    #     # get prediction probabilities for all
    #     proba_mat = self.clf.predict_proba(decision_block_subset)

    #     proba = proba_mat[:, 1]
    #     relative_proba = proba / np.amax(proba)

    #     log_proba = np.log(relative_proba)
    #     logger.info("log relative probabilities:\n%s", log_proba)

    #     # the selection is the highest probability
    #     prediction = int(np.where(proba == np.amax(proba))[0][0])

    #     self.predictions.append(prediction)
    #     self.pred_probas.append(proba_mat)

    #     return Prediction(labels=prediction, probabilities=proba_mat)

    @abstractmethod
    def fit(self):
        """Abstract method to fit classifier

        Returns
        -------
        `None`

        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Prediction:
        """Abstract method to predict with classifier

        X : numpy.ndarray
            3D array where shape = (trials, channels, samples)

        Returns
        -------
        prediction : Prediction
            Results of predict call containing the predicted class labels, and
            optionally the probabilities of the labels (empty list if not possible).

        """
        pass
