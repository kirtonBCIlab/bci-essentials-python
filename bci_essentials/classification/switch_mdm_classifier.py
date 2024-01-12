"""**Switch MDM Classifier **

This is a switch_classifier.
- This means that classification occurs between neutral and one other
label (i.e. Binary classification).
- The produced probabilities between labels are then compared for one
final classification.

**`ToDo`: Missing correct implementation of this classifier**'

"""

# Stock libraries
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn import preprocessing
from pyriemann.classification import MDM

# Import bci_essentials modules and methods
from ..classification.generic_classifier import GenericClassifier, Prediction
from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


# TODO: Missing correct implementation of this classifier
class SwitchMdmClassifier(GenericClassifier):
    """Switch MDM Classifier class (*inherits from GenericClassifier*)."""

    def set_switch_classifier_mdm_settings(
        self,
        n_splits=2,
        rebuild=True,
        random_seed=42,
        n_jobs=1,
        activation_main="relu",
        activation_class="sigmoid",
    ):
        """Set the Switch Classifier MDM settings.

        Parameters
        ----------
        n_splits : int, *optional*
            Number of folds for cross-validation.
            - Default is `2`.
        rebuild : bool, *optional*
            Rebuild the classifier each time. *More description needed*.
            - Default is `True`.
        random_seed : int, *optional*
            Random seed.
            - Default is `42`.
        n_jobs : int, *optional*
            The number of threads to dedicate to this calculation.
        activation_main : str, *optional*
            Activation function for hidden layers.
            - Default is `relu`.
        activation_class : str, *optional*
            Activation function for the output layer.
            - Default is `sigmoid`.

        Returns
        -------
        `None`
            Models created are used in `fit()`.

        """
        self.n_splits = n_splits
        self.cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_seed
        )
        self.rebuild = rebuild

        mdm = MDM(metric=dict(mean="riemann", distance="riemann"), n_jobs=n_jobs)
        self.clf_model = Pipeline([("MDM", mdm)])
        self.clf = Pipeline([("MDM", mdm)])
        # self.clf0and1 = MDM()

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
        X = np.array(self.X)
        y = np.array(self.y)

        # find the number of classes in y there shoud be N + 1, where N is the number of objects in the scene and also the number of classifiers
        self.num_classifiers = len(list(np.unique(self.y))) - 1
        logger.info("Number of classes: %s", self.num_classifiers)

        # make a list to hold all of the classifiers
        self.clfs = []

        # loop through and build the classifiers
        for i in range(self.num_classifiers):
            # take a subset / do spatial filtering
            X = X[:, :, :]  # Does nothing for now

            class_indices = np.logical_or(y == 0, y == (i + 1))
            X_class = X[class_indices, :, :]
            y_class = y[class_indices]

            # Try rebuilding the classifier each time
            if self.rebuild:
                self.next_fit_trial = 0
                # tf.keras.backend.clear_session()

            subX = X_class[self.next_fit_trial :, :, :]
            suby = y_class[self.next_fit_trial :]
            self.next_fit_trial = n_trials

            for train_idx, test_idx in self.cv.split(subX, suby):
                X_train, X_test = subX[train_idx], subX[test_idx]
                y_train, y_test = suby[train_idx], suby[test_idx]

                z_dim, y_dim, x_dim = X_train.shape
                X_train = X_train.reshape(z_dim, x_dim * y_dim)
                scaler_train = preprocessing.StandardScaler().fit(X_train)
                X_train_scaled = scaler_train.transform(X_train)

                logger.info("The shape of X_train_scaled is %s", X_train_scaled.shape)

                z_dim, y_dim, x_dim = X_test.shape
                X_test = X_test.reshape(z_dim, x_dim * y_dim)
                scaler_test = preprocessing.StandardScaler().fit(X_test)
                X_test_scaled = scaler_test.transform(X_test)

                if i == 0:
                    # Compile the model
                    logger.info("\nWorking on first model...")
                    self.clf0and1.compile(
                        # optimizer=Adam(learning_rate=0.001),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"],
                    )
                    # Fit the model
                    self.clf0and1.fit(
                        x=X_train_scaled,
                        y=y_train,
                        batch_size=5,
                        epochs=4,
                        shuffle=True,
                        verbose=2,
                        validation_data=(X_test_scaled, y_test),
                    )  # Need to reshape X_train

                else:
                    logger.info("\nWorking on second model...")
                    # Compile the model
                    self.clf0and2.compile(
                        # optimizer=Adam(learning_rate=0.001),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"],
                    )
                    # Fit the model
                    self.clf0and2.fit(
                        x=X_train_scaled,
                        y=y_train,
                        batch_size=5,
                        epochs=4,
                        shuffle=True,
                        verbose=2,
                        validation_data=(X_test_scaled, y_test),
                    )  # Need to reshape X_train

            # Log performance stats
            # accuracy
            # correct = preds == self.y
            # logger.info("Correct: %s", correct)

            # COMMENTED OUT DUE TO INCOMPLETE IMPLEMENTATION
            """
            self.offline_trial_count = n_trials
            self.offline_trial_counts.append(self.offline_trial_count)
            # accuracy
            accuracy = sum(preds == self.y) / len(preds)
            self.offline_accuracy.append(accuracy)
            logger.info("Accuracy = %s", accuracy)
            # precision
            precision = precision_score(self.y, preds, average="micro")
            self.offline_precision.append(precision)
            logger.info("Precision = %s", precision))
            # recall
            recall = recall_score(self.y, preds, average="micro")
            self.offline_recall.append(recall)
            logger.info("Recall = %s", recall)
            # confusion matrix in command line
            cm = confusion_matrix(self.y, preds)
            self.offline_cm = cm
            logger.info("Confusion matrix:\n%s", cm)
            """

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            3D array where shape = (trials, channels, samples)

        Returns
        -------
        prediction : Prediction
            Results of predict call containing the predicted class labels.  Probabilities
            are not available (empty list).

        """
        # if X is 2D, make it 3D with one as first dimension
        if len(X.shape) < 3:
            X = X[np.newaxis, ...]

        logger.info("The shape of X is %s", X.shape)

        # self.predict0and1 = Sequential(
        #     [
        #         Flatten(),
        #         Dense(units=8, input_shape=(4,), activation="relu"),
        #         Dense(units=16, activation="relu"),
        #         Dense(units=3, activation="sigmoid"),
        #     ]
        # )

        # self.predict0and2 = Sequential(
        #     [
        #         Flatten(),
        #         Dense(units=8, input_shape=(4,), activation="relu"),
        #         Dense(units=16, activation="relu"),
        #         Dense(units=3, activation="sigmoid"),
        #     ]
        # )

        z_dim, y_dim, x_dim = X.shape
        X_predict = X.reshape(z_dim, x_dim * y_dim)
        scaler_train = preprocessing.StandardScaler().fit(X_predict)
        X_predict_scaled = scaler_train.transform(X_predict)

        pred0and1 = self.predict0and1.predict(X_predict_scaled)
        pred0and2 = self.predict0and2.predict(X_predict_scaled)

        final_predictions = np.array([])

        for row1, row2 in zip(pred0and1, pred0and2):
            if row1[0] > row1[1] and row2[0] > row2[2]:
                np.append(final_predictions, 0)
            elif row1[0] > row1[1] and row2[0] < row2[2]:
                np.append(final_predictions, 2)
            elif row1[0] < row1[1] and row2[0] > row2[2]:
                np.append(final_predictions, 1)
            elif row1[0] < row1[1] and row2[0] < row2[2]:
                if row1[1] > row2[2]:
                    np.append(final_predictions, 1)
                else:
                    np.append(final_predictions, 2)

        return Prediction(labels=final_predictions)
