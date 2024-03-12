"""**MI Classifier**

This classifier is used to classify MI data.

"""

# Stock libraries
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

# Import bci_essentials modules and methods
from ..classification.generic_classifier import GenericClassifier, Prediction
from ..channel_selection import channel_selection_by_method
from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


class MiClassifier(GenericClassifier):
    """MI Classifier class (*inherits from `GenericClassifier`*)."""

    def set_mi_classifier_settings(
        self,
        n_splits=5,
        type="TS",
        remove_flats=False,
        whitening=False,
        covariance_estimator="oas",
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
            - Default is `"oas"`.
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
            logger.error("Classifier type not defined")

        if artifact_rejection == "potato":
            logger.error("Potato not implemented")

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

    def fit(self):
        """Fit the model.

        Returns
        -------
        `None`
            Models created used in `predict()`.

        """
        # get dimensions
        n_trials, n_channels, n_samples = self.X.shape

        # do the rest of the training if train_free is false
        self.X = np.array(self.X)

        # Try rebuilding the classifier each time
        if self.rebuild:
            self.next_fit_trial = 0
            self.clf = self.clf_model

        # get temporal subset
        subX = self.X[self.next_fit_trial :, :, :]
        suby = self.y[self.next_fit_trial :]
        self.next_fit_trial = n_trials

        # Init predictions to all false
        preds = np.zeros(n_trials)

        def __mi_kernel(subX, suby):
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
            if self.chs_iterative_selection is True and self.subset is not None:
                initial_subset = self.subset
                logger.info(
                    "Using subset from previous channel selection "
                    + "because iterative selection is TRUE"
                )
            else:
                initial_subset = self.chs_initial_subset

            logger.info("Doing channel selection")
            (
                updated_subset,
                updated_model,
                preds,
                accuracy,
                precision,
                recall,
                results_df,
            ) = channel_selection_by_method(
                __mi_kernel,
                self.X,
                self.y,
                self.channel_labels,  # kernel setup
                self.chs_method,
                self.chs_metric,
                initial_subset,  # wrapper setup
                self.chs_max_time,
                self.chs_min_channels,
                self.chs_max_channels,
                self.chs_performance_delta,  # stopping criterion
                self.chs_n_jobs,
            )

            self.results_df = results_df
            self.subset = updated_subset
            self.clf = updated_model
        else:
            logger.warning("Not doing channel selection")
            self.clf, preds, accuracy, precision, recall = __mi_kernel(subX, suby)

        # Log performance stats

        self.offline_trial_count = n_trials
        self.offline_trial_counts.append(self.offline_trial_count)

        # accuracy
        accuracy = sum(preds == self.y) / len(preds)
        self.offline_accuracy.append(accuracy)
        logger.info("Accuracy = %s", accuracy)

        # precision
        precision = precision_score(self.y, preds, average="micro")
        self.offline_precision.append(precision)
        logger.info("Precision = %s", precision)

        # recall
        recall = recall_score(self.y, preds, average="micro")
        self.offline_recall.append(recall)
        logger.info("Recall = %s", recall)

        # confusion matrix in command line
        cm = confusion_matrix(self.y, preds)
        self.offline_cm = cm
        logger.info("Confusion matrix:\n%s", cm)

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            3D array where shape = (trials, channels, samples)

        Returns
        -------
        prediction : Prediction
            Results of predict call containing the predicted class labels, and
            the probabilities of the labels.

        """
        # if X is 2D, make it 3D with one as first dimension
        if len(X.shape) < 3:
            X = X[np.newaxis, ...]

        subset_X = self.get_subset(X, self.subset, self.channel_labels)

        logger.info("The shape of X is %s", subset_X.shape)

        cov_subset_X = Covariances(estimator=self.covariance_estimator).transform(
            subset_X
        )

        pred = self.clf.predict(cov_subset_X)
        pred_proba = self.clf.predict_proba(cov_subset_X)

        logger.info("Prediction: %s", pred)
        logger.info("Prediction probabilities: %s", pred_proba)

        for i in range(len(pred)):
            self.predictions.append(pred[i])
            self.pred_probas.append(pred_proba[i])

        return Prediction(labels=pred, probabilities=pred_proba)
