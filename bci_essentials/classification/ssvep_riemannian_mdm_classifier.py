"""
**SSVEP Riemannian MDM Classifier**

Classifies SSVEP based on relative band power at the expected
frequencies.

"""

# Stock libraries
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances

# Import bci_essentials modules and methods
from ..classification.generic_classifier import GenericClassifier, Prediction
from ..signal_processing import bandpass
from ..channel_selection import channel_selection_by_method
from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


class SsvepRiemannianMdmClassifier(GenericClassifier):
    """SSVEP Riemannian MDM Classifier class
    (*inherits from GenericClassifier*)

    """

    def set_ssvep_settings(
        self,
        n_splits=3,
        random_seed=42,
        n_harmonics=2,
        f_width=0.2,
        covariance_estimator="oas",
    ):
        """Set the SSVEP settings.

        Parameters
        ----------
        n_splits : int, *optional*
            Number of folds for cross-validation.
            - Default is `3`.
        random_seed : int, *optional*
            Random seed.
            - Default is `42`.
        n_harmonics : int, *optional*
            Number of harmonics to be used for each frequency.
            - Default is `2`.
        f_width : float, *optional*
            Width of frequency bins to be used around the target
            frequencies.
            - Default is `0.2`.
        covariance_estimator : str, *optional*
            Covariance Estimator (see Covariances - pyriemann)
            - Default is `"oas"`.

        Returns
        -------
        `None`
            Models created used in `fit()`.

        """
        # Build the cross-validation split
        self.n_splits = n_splits
        self.cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_seed
        )

        self.rebuild = True

        self.n_harmonics = n_harmonics
        self.f_width = f_width
        self.covariance_estimator = covariance_estimator

        # Use an MDM classifier, maybe there will be other options later
        mdm = MDM(metric=dict(mean="riemann", distance="riemann"), n_jobs=1)
        self.clf_model = Pipeline([("MDM", mdm)])
        self.clf = Pipeline([("MDM", mdm)])

    def get_ssvep_supertrial(
        self,
        X,
        target_freqs,
        fsample,
        f_width=0.4,
        n_harmonics=2,
        covariance_estimator="oas",
    ):
        """Get SSVEP Supertrial.

        Creates the Riemannian Geometry supertrial for SSVEP.

        Parameters
        ----------
        X : numpy.ndarray
            Trials of EEG data.
            3D array containing data with `float` type.

            shape = (`n_trials`,`n_channels`,`n_samples`)
        target_freqs : numpy.ndarray
            Target frequencies for the SSVEP.
        fsample : float
            Sampling rate.
        f_width : float, *optional*
            Width of frequency bins to be used around the target
            frequencies.
            - Default is `0.4`.
        n_harmonics : int, *optional*
            Number of harmonics to be used for each frequency.
            - Default is `2`.
        covarianc_estimator : str, *optional*
            Covariance Estimator (see Covariances - pyriemann)
            - Default is `"oas"`.

        Returns
        -------
        super_X : numpy.ndarray
            Supertrials of X.
            3D array containing data with `float` type.

            shape = (`n_trials`,`n_channels*number of target_freqs`,
            `n_channels*number of target_freqs`)

        """
        n_trials, n_channels, n_samples = X.shape
        n_target_freqs = len(target_freqs)

        super_X = np.zeros(
            [n_trials, n_channels * n_target_freqs, n_channels * n_target_freqs]
        )

        # Create super trial of all trials filtered at all bands
        for trial in range(n_trials):
            for tf, target_freq in enumerate(target_freqs):
                lower_bound = int((n_channels * tf))
                upper_bound = int((n_channels * tf) + n_channels)

                signal = X[trial, :, :]
                for f in range(n_harmonics):
                    if f == 0:
                        filt_signal = bandpass(
                            signal,
                            f_low=target_freq - (f_width / 2),
                            f_high=target_freq + (f_width / 2),
                            order=5,
                            fsample=fsample,
                        )
                    else:
                        filt_signal += bandpass(
                            signal,
                            f_low=(target_freq * (f + 1)) - (f_width / 2),
                            f_high=(target_freq * (f + 1)) + (f_width / 2),
                            order=5,
                            fsample=fsample,
                        )

                cov_mat = Covariances(estimator=covariance_estimator).transform(
                    np.expand_dims(filt_signal, axis=0)
                )

                cov_mat_diag = np.diag(np.diag(cov_mat[0, :, :]))

                super_X[trial, lower_bound:upper_bound, lower_bound:upper_bound] = (
                    cov_mat_diag
                )

        return super_X

    def fit(self):
        """Fit the model.

        Returns
        -------
        `None`
            Models created used in `predict()`.

        """
        # get dimensions
        # X = self.X

        # Convert each trial of X into a SPD of dimensions [n_trials, n_channels*nfreqs, n_channels*nfreqs]
        n_trials, n_channels, n_samples = self.X.shape

        #################
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

        def __ssvep_kernel(subX, suby):
            """SSVEP kernel.

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
                y_train = suby[train_idx]
                # y_train, y_test = suby[train_idx], suby[test_idx]

                # get the covariance matrices for the training set
                X_train_super = self.get_ssvep_supertrial(
                    X_train,
                    self.target_freqs,
                    fsample=256,
                    n_harmonics=self.n_harmonics,
                    f_width=self.f_width,
                    covariance_estimator=self.covariance_estimator,
                )
                X_test_super = self.get_ssvep_supertrial(
                    X_test,
                    self.target_freqs,
                    fsample=256,
                    n_harmonics=self.n_harmonics,
                    f_width=self.f_width,
                    covariance_estimator=self.covariance_estimator,
                )

                # fit the classsifier
                self.clf.fit(X_train_super, y_train)
                preds[test_idx] = self.clf.predict(X_test_super)

            accuracy = sum(preds == self.y) / len(preds)
            precision = precision_score(self.y, preds, average="micro")
            recall = recall_score(self.y, preds, average="micro")

            model = self.clf

            return model, preds, accuracy, precision, recall

        # Check if channel selection is true
        if self.channel_selection_setup:
            logger.info("Doing channel selection")

            (
                updated_subset,
                updated_model,
                preds,
                accuracy,
                precision,
                recall,
            ) = channel_selection_by_method(
                __ssvep_kernel,
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
            )

            logger.info("The optimal subset is: %s", updated_subset)

            self.subset = updated_subset
            self.clf = updated_model
        else:
            logger.warning("Not doing channel selection")
            self.clf, preds, accuracy, precision, recall = __ssvep_kernel(subX, suby)

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

        X = self.get_subset(X)

        logger.info("The shape of X is %s", X.shape)

        X_super = self.get_ssvep_supertrial(
            X,
            self.target_freqs,
            fsample=256,
            n_harmonics=self.n_harmonics,
            f_width=self.f_width,
        )

        pred = self.clf.predict(X_super)
        pred_proba = self.clf.predict_proba(X_super)

        logger.info("Prediction: %s", pred)
        logger.info("Prediction probabilities: %s", pred_proba)

        for i in range(len(pred)):
            self.predictions.append(pred[i])
            self.pred_probas.append(pred_proba[i])

        return Prediction(labels=pred, probabilities=pred_proba)
