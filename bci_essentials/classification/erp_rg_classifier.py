"""**ERP RG Classifier**

This classifier is used to classify ERPs using the Riemannian Geometry
approach.

"""

# Stock libraries
import os
import sys
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
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

from bci_essentials.classification import Generic_classifier

from bci_essentials.signal_processing import lico
from bci_essentials.channel_selection import channel_selection_by_method

# Custom libraries
# - Append higher directory to import bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


class ERP_rg_classifier(Generic_classifier):
    """ERP RG Classifier class (*inherits from `Generic_classifier`*)."""

    def set_p300_clf_settings(
        self,
        n_splits=3,  # number of folds for cross-validation
        lico_expansion_factor=1,  # Linear Combination Oversampling expansion factor is the factor by which the number of ERPs in the training set will be expanded
        oversample_ratio=0,  # traditional oversampling, float from 0.1-1 resulting ratio of erp class to non-erp class, 0 for no oversampling
        undersample_ratio=0,  # traditional undersampling, float from 0.1-1 resulting ratio of erp class to non-erp classs, 0 for no undersampling
        random_seed=42,  # random seed
        covariance_estimator="scm",  # Covarianc estimator, see pyriemann Covariances
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
        covariance_estimator : str, *optional*
            Covariance estimator. See pyriemann Covariances.
            - Default is `"scm"`.

        Returns
        -------
        `None`

        """
        self.n_splits = n_splits
        self.lico_expansion_factor = lico_expansion_factor
        self.oversample_ratio = oversample_ratio
        self.undersample_ratio = undersample_ratio
        self.random_seed = random_seed
        self.covariance_estimator = covariance_estimator

    def add_to_train(self, decision_block, label_idx, print_training=True):
        """Add to training set.

        Parameters
        ----------
        decision_block : numpy.ndarray
            Description of parameter `decision block`.
            If array, state size and type. E.g.
            3D array containing data with `float` type.

            shape = (`1st_dimension`,`2nd_dimension`,`3rd_dimension`)
        label_idx : type
            Description of parameter `label_idx`.
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

        # get a subset
        decision_block = self.get_subset(decision_block)

        # get labels from label_idx
        labels = np.zeros([p])
        labels[label_idx] = 1
        if print_training:
            print(labels)

        # If the classifier has no data then initialize
        if self.X.size == 0:
            self.X = decision_block
            self.y = labels

        # If the classifier already has data then append
        else:
            self.X = np.append(self.X, decision_block, axis=0)
            self.y = np.append(self.y, labels, axis=0)

    def fit(
        self,
        n_splits=2,
        plot_cm=False,
        plot_roc=False,
        lico_expansion_factor=1,
        print_fit=True,
        print_performance=True,
    ):
        """Fit the model.

        Parameters
        ----------
        n_splits : int, *optional*
            Description of parameter `n_splits`.
            - Default is `2`.
        plot_cm : bool, *optional*
            Description of parameter `plot_cm`.
            - Default is `False`.
        plot_roc : bool, *optional*
            Description of parameter `plot_roc`.
            - Default is `False`.
        lico_expansion_factor : int, *optional*
            Description of parameter `lico_expansion_factor`.
            - Default is `1`.
        print_fit : bool, *optional*
            Description of parameter `print_fit`.
            - Default is `True`.
        print_performance : bool, *optional*
            Description of parameter `print_performance`.
            - Default is `True`.

        Returns
        -------
        `None`
            Models created used in `predict()`.

        """
        if print_fit:
            print("Fitting the model using RG")
            print(self.X.shape, self.y.shape)

        # Define the strategy for cross validation
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_seed
        )

        # Define the classifier
        self.clf = make_pipeline(
            XdawnCovariances(estimator=self.covariance_estimator),
            TangentSpace(metric="riemann"),
            LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"),
        )

        # Init predictions to all false
        preds = np.zeros(len(self.y))

        #
        def erp_rg_kernel(X, y):
            """ERP RG kernel.

            Parameters
            ----------
            X : numpy.ndarray
                Description of parameter `X`.
                If array, state size and type. E.g.
                3D array containing data with `float` type.

                shape = (`1st_dimension`,`2nd_dimension`,`3rd_dimension`)
            y : numpy.ndarray
                Description of parameter `y`.
                If array, state size and type. E.g.
                1D array containing data with `int` type.

                shape = (`1st_dimension`,)

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
            for train_idx, test_idx in cv.split(X, y):
                y_train, y_test = y[train_idx], y[test_idx]

                X_train = X[train_idx]
                X_test = X[test_idx]

                # LICO
                if print_fit:
                    print(
                        "Before LICO: Shape X", X_train.shape, "Shape y", y_train.shape
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
                        if print_fit:
                            print("y_train =", y_train)
                if print_fit:
                    print(
                        "After LICO: Shape X", X_train.shape, "Shape y", y_train.shape
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

                if print_fit:
                    print("y_test =", y_test)

                    print(predproba)
                    print(real[0])
                    print(prediction)

            model = self.clf

            accuracy = sum(preds == self.y) / len(preds)
            precision = precision_score(self.y, preds)
            recall = recall_score(self.y, preds)

            return model, preds, accuracy, precision, recall

        # Check if channel selection is true
        if self.channel_selection_setup:
            print("Doing channel selection")
            # print("Initial subset ", self.chs_initial_subset)

            (
                updated_subset,
                updated_model,
                preds,
                accuracy,
                precision,
                recall,
            ) = channel_selection_by_method(
                erp_rg_kernel,
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
                self.chs_output,
            )  # njobs, output messages

            print("The optimal subset is ", updated_subset)

            self.subset = updated_subset
            self.clf = updated_model
        else:
            print("Not doing channel selection")
            self.clf, preds, accuracy, precision, recall = erp_rg_kernel(self.X, self.y)

        # Print performance stats
        # accuracy
        accuracy = sum(preds == self.y) / len(preds)
        self.offline_accuracy = accuracy
        if print_performance:
            print("accuracy = {}".format(accuracy))

        # precision
        precision = precision_score(self.y, preds)
        self.offline_precision = precision
        if print_performance:
            print("precision = {}".format(precision))

        # recall
        recall = recall_score(self.y, preds)
        self.offline_recall = recall
        if print_performance:
            print("recall = {}".format(recall))

        # confusion matrix in command line
        cm = confusion_matrix(self.y, preds)
        self.offline_cm = cm
        if print_performance:
            print("confusion matrix")
            print(cm)

        if plot_cm:
            cm = confusion_matrix(self.y, preds)
            ConfusionMatrixDisplay(cm).plot()
            plt.show()

        if plot_roc:
            print("plotting the ROC...")
            print("just kidding ROC has not been implemented")
