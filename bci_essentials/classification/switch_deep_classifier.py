"""**Switch Deep Classifier**

This is a switch_classifier using a deep neural network implemented
in TensorFlow.
- This means that classification occurs between neutral and one other
label (i.e. Binary classification).
- The produced probabilities between labels are then compared for one
final classification.

**`ToDo`: Missing correct implementation of this classifier**.
The neural networks are not defined (the code blocks are commented out).

"""

# Stock libraries
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing
import tensorflow as tf

from bci_essentials.classification import Generic_classifier

# Custom libraries
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


class Switch_deep_classifier(Generic_classifier):
    """Switch Deep Classifier class
    (*inherits from Generic_classifier*).

    """

    def set_switch_classifier_settings(
        self,
        n_splits=2,
        rebuild=True,
        random_seed=42,
        activation_main="relu",
        activation_class="sigmoid",
    ):
        """Function defines all basic settings for classification.

        Function has 6 parameters and defines two neural networks.
        One that will have weights and another that will not.

        Parameters
        ----------
        n_splits : int, *optional*
            Number of splits for StratifiedKFold.
            - Default is `2`.
        rebuild : bool, *optional*
            Resetting index for each call of `fit()`.
            - Default is `True`.
        random_seed : int, *optional*
            Random seed. Ensures the same output for neural net each run if
            no parameters are changed
            - Default is `42`.
        activation_main : str, *optional*
            Activation function for hidden layers of the neural network.
            - Default is `relu`.
        activation_class : str, *optional*
            Activation function for the final layer of the neural network.
            - Default is `sigmoid`.

        Returns
        -------
        `None`
            Models created are used in `fit()`.

        """
        # Definining activation functions
        self.activation_main = activation_main
        self.activation_class = activation_class

        # Defining training splits
        self.n_splits = n_splits
        self.cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_seed
        )
        self.rebuild = rebuild

        self.random_seed = random_seed

        # Setting random seed for tensorflow so results remain the same for each model
        tf.random.set_seed(random_seed)

        """Defining the neural network:
            self.clf: This is the classifier that will be trained and whose weights will differ. At the end of training the classifier is appended to a list
            self.clf_model: This will remain an unweighted version of the neural network and will be used to reset self.clf"""

        # COMMENTED OUT DUE TO INCOMPLETE IMPLEMENTATION
        """
        self.clf = Sequential(
            [
                Flatten(),
                Dense(units=8, input_shape=(4,), activation=self.activation_main),
                Dense(units=16, activation=self.activation_main),
                Dense(units=3, activation=self.activation_class),
            ]
        )

        self.clf_model = Sequential(
            [
                Flatten(),
                Dense(units=8, input_shape=(4,), activation=self.activation_main),
                Dense(units=16, activation=self.activation_main),
                Dense(units=3, activation=self.activation_class),
            ]
        )
        """

    def fit(self, print_fit=True, print_performance=True):
        """Fitting function for Switch_deep_classifier.

        Function uses the StratifiedKFold() function to split the data and
        then preprocess it using StandardScalar().
        The neural network is then fit and appended to a list before being reset.

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
        # Check for list and correct if needed
        if isinstance(self.X, list):
            print("Error. Self.X should not be a list")
            print("Correcting now...")
            self.X = np.array(self.X)

        # get dimensions
        nwindows, nchannels, nsamples = self.X.shape

        # do the rest of the training if train_free is false
        X = np.array(self.X)
        y = np.array(self.y)

        # list of classifiers
        self.clfs = []

        # Determining number of classes (0, 1, 2 normally)
        self.num_classes = len(np.unique(y))
        print(f"Unique self.y: {np.unique(self.y)}")

        # find the number of classes in y there shoud be N + 1, where N is the number of objects in the scene and also the number of classifiers
        print(f"Number of classes: {self.num_classes}")

        # loop through and build the classifiers. Classification should occur between neutral and an activation state
        for i in range(self.num_classes - 1):
            print("\nStarting on model...")
            # take a subset / do spatial filtering
            X = X[:, :, :]  # Does nothing for now

            # Changing the x array and y array so that their indicies match up and appropriate features are trained with appropraite labels
            # This is so training can be done on 0 vs 1 dataset and 0 vs 2 dataset
            X_class = X[np.logical_or(y == 0, y == (i + 1)), :, :]
            y_class = y[np.logical_or(y == 0, y == (i + 1))]

            X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
                X_class, y_class, test_size=0.15, random_state=self.random_seed
            )

            # Try rebuilding the classifier each time
            if self.rebuild:
                self.next_fit_window = 0

            subX = X_class_train[self.next_fit_window :, :, :]
            suby = y_class_train[self.next_fit_window :]
            self.next_fit_window = nwindows

            preds = np.zeros((nwindows, self.num_classes))
            # preds_multiclass = np.zeros(nwindows)

            for train_idx, test_idx in self.cv.split(subX, suby):
                X_train, X_test = subX[train_idx], subX[test_idx]
                y_train, y_test = suby[train_idx], suby[test_idx]

                # Reshaping the training data makes it easier to fit it to the neural network and other machine learning models
                z_dim, y_dim, x_dim = X_train.shape
                X_train = X_train.reshape(z_dim, x_dim * y_dim)
                # Scaling the data
                scaler_train = preprocessing.StandardScaler().fit(X_train)
                X_train_scaled = scaler_train.transform(X_train)

                # Repeating preprocessing steps done for training data on testing data
                z_dim, y_dim, x_dim = X_test.shape
                X_test = X_test.reshape(z_dim, x_dim * y_dim)
                scaler_test = preprocessing.StandardScaler().fit(X_test)
                X_test_scaled = scaler_test.transform(X_test)

                # Compile the model
                self.clf.compile(
                    # optimizer=Adam(learning_rate=0.001),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )
                # Fit the model
                self.clf.fit(
                    x=X_train_scaled,
                    y=y_train,
                    batch_size=5,
                    epochs=4,
                    shuffle=True,
                    verbose=2,
                    validation_data=(X_test_scaled, y_test),
                )  # Need to reshape X_train
                # preds[test_idx,:] = self.clf.predict(X_test_scaled)

            # Append classifier to list
            self.clfs.append(self.clf)
            # Remove weights on classifer for next run through for loop
            self.clf = self.clf_model

            print("\nFinished model.")

            self.offline_window_count = nwindows
            self.offline_window_counts.append(self.offline_window_count)

            # accuracy
            if print_performance:
                z_dim, y_dim, x_dim = X_class_test.shape
                X_class_test = X_class_test.reshape(z_dim, x_dim * y_dim)
                # Scaling the data
                scaler_train = preprocessing.StandardScaler().fit(X_class_test)
                X_class_test_scaled = scaler_train.transform(X_class_test)

                preds = self.clf.predict(X_class_test_scaled)

                final_preds = np.array([])

                print(f"preds is: {preds}")

                for row in preds:
                    print(f"row is: {row}")
                    if i == 0:
                        if row[0] > row[1]:
                            final_preds = np.append(final_preds, 0)
                        elif row[0] < row[1]:
                            final_preds = np.append(final_preds, 1)
                    elif i == 1:
                        if row[0] > row[2]:
                            final_preds = np.append(final_preds, 0)
                        elif row[0] < row[2]:
                            final_preds = np.append(final_preds, 2)

                accuracy = accuracy_score(y_class_test, final_preds)
                self.offline_accuracy.append(accuracy)

                print(f"final_preds is: {final_preds}")
                print(f"y_class_test is: {y_class_test}")

                print("accuracy = {}".format(accuracy))

            # confusion matrix in command line
            cm = confusion_matrix(y_class_test, final_preds)
            self.offline_cm = cm
            if print_performance:
                print("confusion matrix")
                print(cm)

    def predict(self, X, print_predict):
        """Predict function which preprocesses data and makes prediction(s).

        Function is passed an array of size `(X, 8, 512)` from `bci_data.py`
        where it will predict upon the likelihood of state 1 vs state 2.
        Only works for three states currently.

        Parameters
        ----------
        X : np.ndarray
            An array that will be predicted upon by previously trained
            models.
        print_predict : bool
            Description of parameter `print_predict`.

        Returns
        -------
        final_string
            Predictions formatted as strings for Unity to process it.

        Raises
        ------
        `None`
            Error if there are not an appropriate amount of labels (three) to complete predictions on.

        """

        # if X is 2D, make it 3D with one as first dimension
        if len(X.shape) < 3:
            X = X[np.newaxis, ...]

        print("the shape of X is", X.shape)

        # Reshaping data and preprocessing the same way as done in fit
        z_dim, y_dim, x_dim = X.shape
        X_predict = X.reshape(z_dim, x_dim * y_dim)
        scaler_train = preprocessing.StandardScaler().fit(X_predict)
        X_predict_scaled = scaler_train.transform(X_predict)

        # Final predictions is good once everything is appended - but data needs to be reformatted in a way that Unity understands
        final_predictions = []

        # Make predictions
        print(f"The number of classsifiers in the list are: {len(self.clfs)}")
        for i in range(len(self.clfs)):
            preds = self.clfs[i].predict(X_predict_scaled)
            final_predictions.append(np.ndarray.tolist(preds))

        # This part of predict is about reformatting the data
        iterations = 0
        temp_list = []
        final_preds = []

        # Copying the important values from final_predictions into new list
        for i in final_predictions:
            for sub_list in i:
                temp_list.append(sub_list[iterations + 1])

            iterations += 1
            final_preds.append(temp_list)
            temp_list = []

        """This will format predictions so that unity can understand them.
        However, it only works with two objects right now because of the x and y in zip"""

        try:
            temp_list_new = []
            formatted_preds = []
            for x, y in zip(final_preds[0], final_preds[1]):
                temp_list_new.append(x)
                temp_list_new.append(y)
                formatted_preds.append(temp_list_new)

                temp_list_new = []

            string_list = []

            for preds_list in formatted_preds:
                for some_float in preds_list:
                    string_list.append(str(some_float))

            final_string = ", ".join(string_list)

            print(f"final preds is: {final_preds}")
            print(f"string_preds are: {final_string}")

            return final_string
        except Exception:
            print(
                "Error - there are not an appropriate amount of labels (three) to complete predictions on"
            )
            return None
