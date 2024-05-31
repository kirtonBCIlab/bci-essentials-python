"""**ERP Single Channel Classifier**

This classifier is used to classify ERPs when only a single channel (ex. ear EEG) is available.

"""

# Stock libraries
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Import bci_essentials modules and methods
from .generic_classifier import GenericClassifier, Prediction
from ..signal_processing import lico
from ..channel_selection import channel_selection_by_method
from ..utils.logger import Logger  # Logger wrapper
from ..utils.reduce_to_single_channel import ReduceToSingleChannel

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


class ErpSingleChannelClassifier(GenericClassifier):
    """ERP Single Channel Classifier class (*inherits from `GenericClassifier`*)."""

    def set_p300_clf_settings(
        self,
        n_splits=3,
        lico_expansion_factor=1,
        oversample_ratio=0,
        undersample_ratio=0,
        random_seed=42,
    ):
        """Set P300 Classifier Settings.

        Parameters
        ----------
        n_splits : int, *optional*
            Number of folds for cross-validation.
            - Default is `3`.
        lico_expansion_factor : int, *optional*
            Linear Combination Oversampling expansion factor, which is the
            factor by which the number of ERPs in the training set will be
            expanded.
            - Default is `1`.
        oversample_ratio : float, *optional*
            Traditional oversampling. Range is from from 0.1-1 resulting
            from the ratio of erp to non-erp class. 0 for no oversampling.
            - Default is `0`.
        undersample_ratio : float, *optional*
            Traditional undersampling. Range is from from 0.1-1 resulting
            from the ratio of erp to non-erp class. 0 for no undersampling.
            - Default is `0`.
        random_seed : int, *optional*
            Random seed.
            - Default is `42`.

        Returns
        -------
        `None`

        """
        self.n_splits = n_splits
        self.lico_expansion_factor = lico_expansion_factor
        self.oversample_ratio = oversample_ratio
        self.undersample_ratio = undersample_ratio
        self.random_seed = random_seed

    def fit(
        self,
        n_splits=2,
        plot_cm=False,
        plot_roc=False,
        lico_expansion_factor=1,
    ):
        """Fit the model.

        Parameters
        ----------
        n_splits : int, *optional*
            Number of folds for cross validation.
            E.g. how many parts the dataset is divided into and trained/validated.
            - Default is `2`.
        plot_cm : bool, *optional*
            Whether to plot the confusion matrix during training.
            - Default is `False`.
        plot_roc : bool, *optional*
            Whether to plot the ROC curve during training.
            - Default is `False`.
        lico_expansion_factor : int, *optional*
            Linear combination oversampling expansion factor.
            Determines the number of ERPs in the training set that will be expanded.
            Higher value increases the oversampling, generating more synthetic
            samples for the minority class.
            - Default is `1`.

        Returns
        -------

        `None`
            Models created used in `predict()`.

        """

        logger.info("Fitting the model using sLDA")
        logger.info("X shape: %s", self.X.shape)
        logger.info("y shape: %s", self.y.shape)

        # Define the strategy for cross validation
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_seed
        )

        # Define the classifier
        self.clf = make_pipeline(
            ReduceToSingleChannel(),
            LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"),
        )

        # Init predictions to all false
        preds = np.zeros(len(self.y))

        #
        def __erp_single_channel_kernel(X, y):
            """ERP Single Channel kernel.

            Parameters
            ----------
            X : numpy.ndarray
                Input features (ERP data) for training.
                3D numpy array with shape = (`n_trials`, `n_channels`, `n_samples`).
                E.g. (100, 1, 1000) for 100 trials, 1 channel and 1000 samples.

            y : numpy.ndarray
                Target labels corresponding to the input features in `X`.
                1D numpy array with shape (n_trails, ).
                Each label indicates the class of the corresponding trial in `X`.
                E.g. (100, ) for 100 trials.


            Returns
            -------
            model : classifier
                The trained classification model.
            preds : numpy.ndarray
                The predictions from the model.
                1D array with the same shape as `y`.
            accuracy : float
                The accuracy of the trained classification model.
            precision : float
                The precision of the trained classification model.
            recall : float
                The recall of the trained classification model.

            """
            print("X shape: ", X.shape)

            for train_idx, test_idx in cv.split(X, y):
                y_train, y_test = y[train_idx], y[test_idx]

                X_train, X_test = X[train_idx], X[test_idx]

                # LICO
                logger.debug(
                    "Before LICO:\n\tShape X: %s\n\tShape y: %s",
                    X_train.shape,
                    y_train.shape,
                )

                if sum(y_train) > 2:
                    if lico_expansion_factor > 1:
                        X_train, y_train = lico(
                            X_train,
                            y_train,
                            expansion_factor=lico_expansion_factor,
                            sum_num=2,
                            shuffle=False,
                        )
                        logger.debug("y_train = %s", y_train)

                logger.debug(
                    "After LICO:\n\tShape X: %s\n\tShape y: %s",
                    X_train.shape,
                    y_train.shape,
                )

                # Oversampling
                if self.oversample_ratio > 0:
                    p_count = sum(y_train)
                    n_count = len(y_train) - sum(y_train)

                    num_to_add = int(
                        np.floor((self.oversample_ratio * n_count) - p_count)
                    )

                    # Add num_to_add random selections from the positive
                    true_X_train = X_train[y_train == 1]

                    len_X_train = len(true_X_train)

                    for s in range(num_to_add):
                        to_add_X = true_X_train[random.randrange(0, len_X_train), :, :]

                        X_train = np.append(X_train, to_add_X[np.newaxis, :], axis=0)
                        y_train = np.append(y_train, [1], axis=0)

                # Undersampling
                if self.undersample_ratio > 0:
                    p_count = sum(y_train)
                    n_count = len(y_train) - sum(y_train)

                    num_to_remove = int(
                        np.floor(n_count - (p_count / self.undersample_ratio))
                    )

                    ind_range = np.arange(len(y_train))
                    ind_list = list(ind_range)
                    to_remove = []

                    # Remove num_to_remove random selections from the negative
                    false_ind = list(ind_range[y_train == 0])

                    for s in range(num_to_remove):
                        # select a random value from the list of false indices
                        remove_at = false_ind[random.randrange(0, len(false_ind))]

                        # remove that value from the false ind list
                        false_ind.remove(remove_at)

                        # add the index to be removed to a list
                        to_remove.append(remove_at)

                    remaining_ind = ind_list
                    for i in range(len(to_remove)):
                        remaining_ind.remove(to_remove[i])

                    X_train = X_train[remaining_ind, :, :]
                    y_train = y_train[remaining_ind]

                self.clf.fit(X_train, y_train)
                preds[test_idx] = self.clf.predict(X_test)
                predproba = self.clf.predict_proba(X_test)

                # Use pred proba to show what would be predicted
                predprobs = predproba[:, 1]
                real = np.where(y_test == 1)

                # TODO handle exception where two probabilities are the same
                prediction = int(np.where(predprobs == np.amax(predprobs))[0][0])

                logger.debug("y_test = %s", y_test)
                logger.debug("predproba = %s", predproba)
                logger.debug("real = %s", real[0])
                logger.debug("prediction = %s", prediction)

            model = self.clf

            accuracy = sum(preds == self.y) / len(preds)
            precision = precision_score(self.y, preds)
            recall = recall_score(self.y, preds)

            return model, preds, accuracy, precision, recall

        # Check if channel selection is true
        if self.channel_selection_setup:
            logger.info("Doing channel selection")
            logger.debug("Initial subset: %s", self.chs_initial_subset)

            (
                updated_subset,
                updated_model,
                preds,
                accuracy,
                precision,
                recall,
                results_df,
            ) = channel_selection_by_method(
                __erp_single_channel_kernel,
                self.X,
                self.y,
                self.channel_labels,  # kernel setup
                self.chs_method,
                self.chs_metric,
                self.chs_initial_subset,  # wrapper setup
                self.chs_max_time,
                self.chs_min_channels,
                self.chs_max_channels,
                self.chs_performance_delta,  # stopping criterion
                self.chs_n_jobs,
            )  # njobs, output messages

            logger.info("The optimal subset is %s", updated_subset)

            self.results_df = results_df
            self.subset = updated_subset
            self.clf = updated_model
        else:
            logger.warning("Not doing channel selection")
            self.clf, preds, accuracy, precision, recall = __erp_single_channel_kernel(
                self.X, self.y
            )

        # Log performance stats
        # accuracy
        accuracy = sum(preds == self.y) / len(preds)
        self.offline_accuracy = accuracy
        logger.info("Accuracy = %s", accuracy)

        # precision
        precision = precision_score(self.y, preds)
        self.offline_precision = precision
        logger.info("Precision = %s", precision)

        # recall
        recall = recall_score(self.y, preds)
        self.offline_recall = recall
        logger.info("Recall = %s", recall)

        # confusion matrix in command line
        cm = confusion_matrix(self.y, preds)
        self.offline_cm = cm
        logger.info("Confusion matrix:\n%s", cm)

        if plot_cm:
            cm = confusion_matrix(self.y, preds)
            ConfusionMatrixDisplay(cm).plot()
            plt.show()

        if plot_roc:
            logger.info("Plotting the ROC...")
            logger.error("Just kidding ROC has not been implemented")

    def predict(self, X):
        """Predict the class of the data (Unused in this classifier)

        Parameters
        ----------
        X : numpy.ndarray
            3D array where shape = (n_trials, n_channels, n_samples)

        Returns
        -------
        prediction : Prediction
            Empty Predict object

        """

        return Prediction()
