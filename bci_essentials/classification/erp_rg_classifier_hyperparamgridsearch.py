"""**ERP RG Classifier**

This classifier is used to classify ERPs using the Riemannian Geometry
approach.

"""

# Stock libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.pipeline import Pipeline
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import XdawnCovariances
from pyriemann.channelselection import FlatChannelRemover
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Import bci_essentials modules and methods
from .generic_classifier import (
    GenericClassifier,
    Prediction,
)
from ..signal_processing import lico, random_oversampling, random_undersampling
from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


class ErpRgClassifierHyperparamGridSearch(GenericClassifier):
    """ERP RG Classifier with hyperparameter grid search
    class (*inherits from `GenericClassifier`*)."""

    def set_p300_clf_settings(
        self,
        n_splits=3,
        resampling_method=None,
        lico_expansion_factor=1,
        oversample_ratio=0,
        undersample_ratio=0,
        random_seed=42,
        remove_flats=True,
    ):
        """Set P300 Classifier Settings.

        Parameters
        ----------
        n_splits : int, *optional*
            Number of folds for cross-validation.
            - Default is `3`.
        resampling_method : str, *optional*, None
            Resampling method to use ["lico", "oversample", "undersample"].
            Default is None.
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
        remove_flats : bool, *optional*
            Whether to remove flat channels.
            - Default is `True`.

        Returns
        -------
        `None`

        """
        self.n_splits = n_splits
        self.resampling_method = resampling_method
        self.lico_expansion_factor = lico_expansion_factor
        self.oversample_ratio = oversample_ratio
        self.undersample_ratio = undersample_ratio
        self.random_seed = random_seed

        # # Create steps list with proper formatting
        steps = []
        if remove_flats:
            steps.append(("remove_flats", FlatChannelRemover()))

        steps.extend(
            [
                ("xdawn", XdawnCovariances()),
                ("tangent", TangentSpace()),
                ("lda", LinearDiscriminantAnalysis()),
            ]
        )

        # Create pipeline
        self.clf = Pipeline(steps)

        # Hyperparameters to be optimized
        self.param_grid = {
            "xdawn__nfilter": [2, 3, 4],
            "xdawn__estimator": ["oas", "lwf"],
            "tangent__metric": ["riemann"],
            "lda__solver": ["lsqr", "eigen"],
            "lda__shrinkage": np.linspace(0.5, 0.9, 5),
        }

    def fit(
        self,
        plot_cm=False,
        plot_roc=False,
    ):
        """Fit the model.

        Parameters
        ----------
        plot_cm : bool, *optional*
            Whether to plot the confusion matrix during training.
            - Default is `False`.
        plot_roc : bool, *optional*
            Whether to plot the ROC curve during training.
            - Default is `False`.

        Returns
        -------
        `None`
            Models created used in `predict()`.

        """
        logger.info("Fitting the model using RG")
        logger.info("X shape: %s", self.X.shape)
        logger.info("y shape: %s", self.y.shape)

        # Resample data if needed
        self.X, self.y = self.__resample_data()

        # Optimize hyperparameters with cross-validation
        self.__optimize_hyperparameters()

        # Fit the model with the complete dataset and optimized hyperparameters
        self.clf.fit(self.X, self.y)

        # Get predictions for final model
        y_pred = self.clf.predict(self.X)
        y_pred_proba = self.clf.predict_proba(self.X)[:, 1]

        # Calculate training metrics of final model
        self.offline_accuracy = sum(y_pred == self.y) / len(self.y)
        self.offline_precision = precision_score(self.y, y_pred)
        self.offline_recall = recall_score(self.y, y_pred)

        try:
            roc_auc = roc_auc_score(self.y, y_pred_proba)
            logger.info(f"ROC AUC Score: {roc_auc:0.3f}")
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC score: {e}")

        # Display training confusion matrix
        self.offline_cm = confusion_matrix(self.y, y_pred)
        if plot_cm:
            disp = ConfusionMatrixDisplay(confusion_matrix=self.offline_cm)
            disp.plot()
            plt.title("Training confusion matrix")

        if plot_roc:
            # TODO Implementation missing
            pass

        # Log training metrics
        logger.info("Final model training performance metrics:")
        logger.info(f"Accuracy: {self.offline_accuracy:0.3f}")
        logger.info(f"Precision: {self.offline_precision:0.3f}")
        logger.info(f"Recall: {self.offline_recall:0.3f}")
        logger.info(f"Confusion Matrix:\n{self.offline_cm}")

    def predict(self, X):
        """Predict the class of the data

        Parameters
        ----------
        X : numpy.ndarray
            3D array where shape = (n_epochs, n_channels, n_samples)

        Returns
        -------
        prediction : Prediction
            Predict object. Contains the predicted labels and and the probability.
            Because this classifier chooses the P300 object with the highest posterior probability,
            the probability is only the posterior probability of the chosen object.

        """

        subset_X = self.get_subset(X, self.subset, self.channel_labels)

        # Get posterior probability for each target
        posterior_prob = self.clf.predict_proba(subset_X)[:, 1]

        label = [int(np.argmax(posterior_prob))]
        probability = [np.max(posterior_prob)]

        return Prediction(label, probability)

    # TODO implement additional resampling methods, JIRA ticket: B4K-342
    def __resample_data(self):
        """Resample data based on the selected method"""

        X_resampled = self.X.copy()
        y_resampled = self.y.copy()

        try:
            if (self.resampling_method == "lico") and (self.lico_expansion_factor > 1):
                [X_resampled, y_resampled] = lico(
                    self.X, self.y, self.lico_expansion_factor
                )
                pass

            elif (self.resampling_method == "oversample") and (
                self.oversample_ratio > 0
            ):
                [X_resampled, y_resampled] = random_oversampling(
                    self.X, self.y, self.oversample_ratio
                )
                pass

            elif (self.resampling_method == "undersample") and (
                self.undersample_ratio > 0
            ):
                [X_resampled, y_resampled] = random_undersampling(
                    self.X, self.y, self.undersample_ratio
                )
                pass

            logger.info(f"Resampling  with {self.resampling_method} done")
            logger.info(f"X_resampled shape: {X_resampled.shape}")
            logger.info(f"y_resampled shape: {y_resampled.shape}")

        except Exception as e:
            logger.error(
                f"{self.resampling_method.capitalize()} resampling method failed"
            )
            logger.error(e)

        return X_resampled, y_resampled

    def __optimize_hyperparameters(self):
        """Optimize hyperparameters with cross-validation using brute force grid search

        Returns
        -------
        `None`
            Model with best hyperparameters to be used in `predict()`.

        """

        # Perform cross-validation
        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_seed
        )

        # Create custom scorer function
        custom_scorer = make_scorer(
            self._valid_roc_auc, response_method="predict_proba", greater_is_better=True
        )

        # Create GridSearchCV object
        grid_search = GridSearchCV(
            estimator=self.clf,
            param_grid=self.param_grid,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            scoring=custom_scorer,
            refit=True,
        )

        # Start grid search optimization
        logger.info("Starting grid search optimization...")
        grid_search.fit(self.X, self.y)

        # Get best parameters and score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Update classifier with best parameters
        self.clf.set_params(**best_params)
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best CV score: {best_score:0.3f}")

    def _valid_roc_auc(self, y_true, y_pred, **kwargs):
        """Calculate the ROC AUC score for the classifier.
        This method is used because the stock `roc_auc_score` function
        does not handle the case where one class is missing in the fold.
        This method will return 0.5 in that case.

        Parameters
        ----------
        y_true : numpy.ndarray
            True labels.
        y_pred : numpy.ndarray
            Predicted labels.
        **kwargs : dict
            Additional keyword arguments passed by make_scorer.

        Returns
        -------
        roc_auc : float
            ROC AUC score.

        """
        try:
            # Check if we have both classes in the fold
            if len(np.unique(y_true)) < 2:
                logger.warning("Fold contains only one class")
                return 0.5

            return roc_auc_score(y_true, y_pred)

        except Exception as e:
            logger.warning(f"ROC AUC calculation failed: {e}")
            return 0.5
