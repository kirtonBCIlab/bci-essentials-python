"""**MI Classifier**

This classifier is used to classify MI data.

"""


# Stock libraries
import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pyriemann.preprocessing import Whitening
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM, TSclassifier
from pyriemann.channelselection import FlatChannelRemover, ElectrodeSelection

from bci_essentials.classification import Generic_classifier

from bci_essentials.channel_selection import channel_selection_by_method

# Custom libraries
# - Append higher directory to import bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


class MI_classifier(Generic_classifier):
    """MI Classifier class (*inherits from `Generic_classifier`*)."""

    def set_mi_classifier_settings(
        self,
        n_splits=5,
        type="TS",
        remove_flats=False,
        whitening=False,
        covariance_estimator="scm",
        artifact_rejection="none",
        channel_selection="none",
        pred_threshold=0.5,
        random_seed=42,
        n_jobs=1,
    ):
        """Set MI classifier settings.

        Parameters
        ----------
        n_splits : int, *optional*
            Number of folds for cross-validation.
            - Default is `5`.
        type : str, *optional*
            Description of parameter `type`.
            - Default is `"TS"`.
        remove_flats : bool, *optional*
            Description of parameter `remove_flats`.
            - Default is `False`.
        whitening : bool, *optional*
            Description of parameter `whitening`.
            - Default is `False`.
        covariance_estimator : str, *optional*
            Covariance estimator. See pyriemann Covariances.
            - Default is `"scm"`.
        artifact_rejection : str, *optional*
            Description of parameter `artifact_rejection`.
            - Default is `"none"`.
        channel_selection : str, *optional*
            Description of parameter `channel_selection`.
            - Default is `"none"`.
        pred_threshold : float, *optional*
            Description of parameter `pred_threshold`.
            - Default is `0.5`.
        random_seed : int, *optional*
            Random seed.
            - Default is `42`.
        n_jobs : int, *optional*
            The number of threads to dedicate to this calculation.
            - Default is `1`.

        Returns
        -------
        `None`
            Models created are used in `fit()`.

        """
        # Build the cross-validation split
        self.n_splits = n_splits
        self.cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_seed
        )

        self.covariance_estimator = covariance_estimator

        # Shrinkage LDA
        if type == "sLDA":
            slda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
            self.clf_model = Pipeline([("Shrinkage LDA", slda)])
            self.clf = Pipeline([("Shrinkage LDA", slda)])

        # Random Forest
        elif type == "RandomForest":
            rf = RandomForestClassifier()
            self.clf_model = Pipeline([("Random Forest", rf)])
            self.clf = Pipeline([("Random Forest", rf)])

        # Tangent Space Logistic Regression
        elif type == "TS":
            ts = TSclassifier()
            self.clf_model = Pipeline([("Tangent Space", ts)])
            self.clf = Pipeline([("Tangent Space", ts)])

        # Minimum Distance to Mean
        elif type == "MDM":
            mdm = MDM(metric=dict(mean="riemann", distance="riemann"), n_jobs=n_jobs)
            self.clf_model = Pipeline([("MDM", mdm)])
            self.clf = Pipeline([("MDM", mdm)])

        # CSP + Logistic Regression (REQUIRES MNE CSP)
        # elif type == "CSP-LR":
        #     lr = LogisticRegression()
        #     self.clf_model = Pipeline([('CSP', csp), ('LogisticRegression', lr)])
        #     self.clf = Pipeline([('CSP', csp), ('LogisticRegression', lr)])

        else:
            print("Classifier type not defined")

        if artifact_rejection == "potato":
            print("Potato not implemented")

        if whitening:
            self.clf_model.steps.insert(0, ["Whitening", Whitening()])
            self.clf.steps.insert(0, ["Whitening", Whitening()])

        if channel_selection == "riemann":
            rcs = ElectrodeSelection()
            self.clf_model.steps.insert(0, ["Channel Selection", rcs])
            self.clf.steps.insert(0, ["Channel Selection", rcs])

        if remove_flats:
            rf = FlatChannelRemover()
            self.clf_model.steps.insert(0, ["Remove Flat Channels", rf])
            self.clf.steps.insert(0, ["Remove Flat Channels", rf])

        # Threshold
        self.pred_threshold = pred_threshold

        # Rebuild from scratch with each training
        self.rebuild = True

    def fit(self, print_fit=True, print_performance=True):
        """Fit the model.

        Parameters
        ----------
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
        # get dimensions
        nwindows, nchannels, nsamples = self.X.shape

        # do the rest of the training if train_free is false
        self.X = np.array(self.X)

        # Try rebuilding the classifier each time
        if self.rebuild:
            self.next_fit_window = 0
            self.clf = self.clf_model

        # get temporal subset
        subX = self.X[self.next_fit_window :, :, :]
        suby = self.y[self.next_fit_window :]
        self.next_fit_window = nwindows

        # Init predictions to all false
        preds = np.zeros(nwindows)

        def mi_kernel(subX, suby):
            """MI kernel.

            Parameters
            ----------
            subX : numpy.ndarray
                Description of parameter `subX`.
                If array, state size and type. E.g.
                3D array containing data with `float` type.

                shape = (`1st_dimension`,`2nd_dimension`,`3rd_dimension`)
            suby : numpy.ndarray
                Description of parameter `suby`.
                If array, state size and type. E.g.
                1D array containing data with `int` type.

                shape = (`1st_dimension`,)
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
            for train_idx, test_idx in self.cv.split(subX, suby):
                self.clf = self.clf_model

                X_train, X_test = subX[train_idx], subX[test_idx]
                # y_test not implemented
                y_train = suby[train_idx]

                # get the covariance matrices for the training set
                X_train_cov = Covariances(
                    estimator=self.covariance_estimator
                ).transform(X_train)
                X_test_cov = Covariances(estimator=self.covariance_estimator).transform(
                    X_test
                )

                # fit the classsifier
                self.clf.fit(X_train_cov, y_train)
                preds[test_idx] = self.clf.predict(X_test_cov)

            accuracy = sum(preds == self.y) / len(preds)
            precision = precision_score(self.y, preds, average="micro")
            recall = recall_score(self.y, preds, average="micro")

            model = self.clf

            return model, preds, accuracy, precision, recall

        # Check if channel selection is true
        if self.channel_selection_setup:
            print("Doing channel selection")

            (
                updated_subset,
                updated_model,
                preds,
                accuracy,
                precision,
                recall,
            ) = channel_selection_by_method(
                mi_kernel,
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
            )
            print("The optimal subset is ", updated_subset)

            self.subset = updated_subset
            self.clf = updated_model
        else:
            print("Not doing channel selection")
            self.clf, preds, accuracy, precision, recall = mi_kernel(subX, suby)

        # Print performance stats

        self.offline_window_count = nwindows
        self.offline_window_counts.append(self.offline_window_count)

        # accuracy
        accuracy = sum(preds == self.y) / len(preds)
        self.offline_accuracy.append(accuracy)
        if print_performance:
            print("accuracy = {}".format(accuracy))

        # precision
        precision = precision_score(self.y, preds, average="micro")
        self.offline_precision.append(precision)
        if print_performance:
            print("precision = {}".format(precision))

        # recall
        recall = recall_score(self.y, preds, average="micro")
        self.offline_recall.append(recall)
        if print_performance:
            print("recall = {}".format(recall))

        # confusion matrix in command line
        cm = confusion_matrix(self.y, preds)
        self.offline_cm = cm
        if print_performance:
            print("confusion matrix")
            print(cm)

    def predict(self, X, print_predict=True):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Description of parameter `X`.
            If array, state size and type. E.g.
            3D array containing data with `float` type.

            shape = (`1st_dimension`,`2nd_dimension`,`3rd_dimension`)
        print_predict : bool, *optional*
            Description of parameter `print_predict`.
            - Default is `True`.

        Returns
        -------
        pred : numpy.ndarray
            The predicted class labels.

            shape = (`1st_dimension`,)

        """
        # if X is 2D, make it 3D with one as first dimension
        if len(X.shape) < 3:
            X = X[np.newaxis, ...]

        X = self.get_subset(X)

        if print_predict:
            print("the shape of X is", X.shape)

        X_cov = Covariances(estimator=self.covariance_estimator).transform(X)

        pred = self.clf.predict(X_cov)
        pred_proba = self.clf.predict_proba(X_cov)

        if print_predict:
            print(pred)
            print(pred_proba)

        for i in range(len(pred)):
            self.predictions.append(pred[i])
            self.pred_probas.append(pred_proba[i])

        return pred
