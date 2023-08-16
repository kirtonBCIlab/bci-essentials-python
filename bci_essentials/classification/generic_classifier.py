"""**Generic classifier class for BCI Essentials**

Used as Parent classifier class for other classifiers.

"""

# Stock libraries
import numpy as np


class Generic_classifier:
    """The base generic classifier class for other classifiers."""

    def __init__(self, training_selection=0, subset=[]):
        """Initializes `Generic_classifier` class.

        Parameters
        ----------
        training_selection : type, *optional*
            Description of parameter `training_selection`.
            - Default is `0`.
        subset : list of `type`, *optional*
            Description of parameter `subset`.
            - Default is `[]`.

        Attributes
        ----------
        X : numpy.ndarray
            Description of attribute `X`.
            If array, state size and type. E.g.
            3D array containing data with `float` type.

            shape = (`1st_dimension`,`2nd_dimension`,`3rd_dimension`)
            - Initial value is `np.ndarray([0])`.
        y : numpy.ndarray
            Description of attribute `y`.
            If array, state size and type. E.g.
            1D array containing data with `int` type.

            shape = (`1st_dimension`,)
            - Initial value is `np.ndarray([0])`.
        subset_defined : bool
            Description of attribute `subset_defined`.
            - Initial value is `False`.
        subset : list of `type`
            Description of attribute `subset`.
            - Initial value is parameter `subset`.
        channel_labels : list of `str`
            Description of attribute `channel_labels`.
            - Initial value is `[]`.
        channel_selection_setup : bool
            Description of attribute `channel_selection_setup`.
            - Initial value is `False`.
        offline_accuracy : list of `float`
            Description of attribute `offline_accuracy`.
            - Initial value is `[]`.
        offline_precision : list of `float`
            Description of attribute `offline_precision`.
            - Initial value is `[]`.
        offline_recall : list of `float`
            Description of attribute `offline_recall`.
            - Initial value is `[]`.
        offline_window_count : int
            Description of attribute `offline_window_count`.
            - Initial value is `0`.
        offline_window_counts : list of `int`
            Description of attribute `offline_window_counts`.
            - Initial value is `[]`.
        next_fit_window : int
            Description of attribute `next_fit_window`.
            - Initial value is `0`.
        predictions : list of `type`
            Description of attribute `predictions`.
            - Initial value is `[]`.
        pred_probas : list of `float`
            Description of attribute `pred_probas`.
            - Initial value is `[]`.

        """
        print("initializing the classifier")
        self.X = np.ndarray([0])
        """@private (This is just for the API docs, to avoid double listing."""
        self.y = np.ndarray([0])
        """@private (This is just for the API docs, to avoid double listing."""

        #
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
        self.offline_window_count = 0
        """@private (This is just for the API docs, to avoid double listing."""
        self.offline_window_counts = []
        """@private (This is just for the API docs, to avoid double listing."""

        # For iterative fitting,
        self.next_fit_window = 0
        """@private (This is just for the API docs, to avoid double listing."""

        # Keep track of predictions
        self.predictions = []
        """@private (This is just for the API docs, to avoid double listing."""
        self.pred_probas = []
        """@private (This is just for the API docs, to avoid double listing."""

    def get_subset(self, X=[]):
        """Get a subset of X according to labels or indices.

        Parameters
        ----------
        X : numpy.ndarray, *optional*
            3D array containing data with `float` type.

            shape = (`N_windows`,`M_channels`,`P_samples`)
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

            shape = (`N_windows`,`M_channels`,`P_samples`)

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
                print("Using subset indices")

                subset_indices = self.subset

            # Or channel labels
            if type(self.subset[0]) is str:
                print("Using channel labels and subset labels")

                # Replace indices with those described by labels
                for sl in self.subset:
                    subset_indices.append(self.channel_labels.index(sl))

            # Return for the given indices
            try:
                # nwindows, nchannels, nsamples = self.X.shape

                if X == []:
                    new_X = self.X[:, subset_indices, :]
                    self.X = new_X
                else:
                    new_X = X[:, subset_indices, :]
                    X = new_X
                    return X

            except Exception:
                # nchannels, nsamples = self.X.shape
                if X == []:
                    new_X = self.X[subset_indices, :]
                    self.X = new_X

                else:
                    new_X = X[subset_indices, :]
                    X = new_X
                    return X

        # notify if failed
        except Exception:
            print("something went wrong, no subset taken")
            return X

    def setup_channel_selection(
        self,
        method="SBS",
        metric="accuracy",
        initial_channels=[],  # wrapper setup
        max_time=999,
        min_channels=1,
        max_channels=999,
        performance_delta=0.001,  # stopping criterion
        n_jobs=1,
        print_output="silent",
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
        initial_channels : type, *optional*
            Description of parameter `initial_channels`.
            - Default is `[]`.
        max_time : type, *optional*
            Description of parameter `max_time`.
            - Default is `999`.
        min_channels : type, *optional*
            Description of parameter `min_channels`.
            - Default is `1`.
        max_channels : type, *optional*
            Description of parameter `max_channels`.
            - Default is `999`.
        performance_delta : type, *optional*
            Description of parameter `performance_delta`.
            - Default is `0.001`.
        n_jobs : type, *optional*
            The number of threads to dedicate to this calculation.
            - Default is `1`.
        print_output : type, *optional*
            Description of parameter `print_output`.
            - Default is `"silent"`.

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
        self.chs_n_jobs = n_jobs  # number of threads
        self.chs_max_time = max_time  # max time in seconds
        self.chs_min_channels = min_channels  # minimum number of channels
        self.chs_max_channels = max_channels  # maximum number of channels
        self.chs_performance_delta = performance_delta  # smallest performance increment to justify continuing search
        self.chs_output = print_output  # output setting, silent, final, or verbose

        self.channel_selection_setup = True

    # add training data, to the training set using a decision block and a label
    def add_to_train(
        self, decision_block, labels, num_options=0, meta=[], print_training=True
    ):
        """Add training data to the training set using a decision block
        and a label.

        Parameters
        ----------
        decision block : type
            Description of parameter `decision block`.
            If array, state size and type. E.g.
            3D array containing data with `float` type.

            shape = (`1st_dimension`,`2nd_dimension`,`3rd_dimension`)
        labels : type
            Description of parameter `labels`.
        num_options : type, *optional*
            Description of parameter `num_options`.
            - Default is `0`.
        meta : type, *optional*
            Description of parameter `meta`.
            - Default is `[]`.
        print_training : bool, *optional*
            Description of parameter `print_training`.
            - Default is `True`.

        Returns
        -------
        `None`

        """
        if print_training:
            print("adding to training set")
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

    # predict a label based on a decision block
    def predict_decision_block(self, decision_block, print_predict=True):
        """Predict a label based on a decision block.

        Parameters
        ----------
        decision block : type
            Description of parameter `decision block`.
            If array, state size and type. E.g.
            3D array containing data with `float` type.

            shape = (`first_dimension`,`second_dimension`,`third_dimension`)
        print_predict : type, *optional*
            Description of parameter `print_predict`.
            - Default is `True`.

        Returns
        -------
        prediction : type
            Description of returned object.

        """
        decision_block = self.get_subset(decision_block)

        if print_predict:
            print("making a prediction")

        # get prediction probabilities for all
        proba_mat = self.clf.predict_proba(decision_block)

        proba = proba_mat[:, 1]

        relative_proba = proba / np.amax(proba)

        log_proba = np.log(relative_proba)

        if print_predict:
            print("log relative probabilities")
            print(log_proba)

        # the selection is the highest probability

        prediction = int(np.where(proba == np.amax(proba))[0][0])

        self.predictions.append(prediction)
        self.pred_probas.append(proba_mat)

        return prediction

    def fit(self, **kwargs):
        """Abstract method to fit classifier

        Parameters
        ----------
        \*\*kwargs : dict, *optional*
            Description of extra arguments to pass to the method.

        Returns
        -------
        `None`

        """
        return None

    def predict(self, **kwargs):
        """Abstract method to predict with classifier

        Parameters
        ----------
        \*\*kwargs : dict, *optional*
            Description of extra arguments to pass to the method.

        Returns
        -------
        `None`

        """
        return None
