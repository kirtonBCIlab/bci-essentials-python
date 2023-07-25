# Stock libraries
import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn import preprocessing
from pyriemann.classification import MDM

# Custom libraries
# - Append higher directory to import bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))
from classification.generic_classifier import Generic_classifier
from bci_essentials.visuals import *
from bci_essentials.signal_processing import *
from bci_essentials.channel_selection import *

# TODO: Missing correct implementation of this classifier
class Switch_mdm_classifier(Generic_classifier):
    def set_switch_classifier_mdm_settings(self, n_splits = 2, rebuild = True, random_seed = 42, n_jobs=1, activation_main = 'relu', activation_class = 'sigmoid'):

        self.n_splits = n_splits
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        self.rebuild = rebuild

        mdm = MDM(metric=dict(mean='riemann', distance='riemann'), n_jobs = n_jobs)
        self.clf_model = Pipeline([("MDM", mdm)])
        self.clf = Pipeline([("MDM", mdm)])
        # self.clf0and1 = MDM()


    def fit(self, print_fit=True, print_performance=True):
        # get dimensions
        nwindows, nchannels, nsamples = self.X.shape 

        # do the rest of the training if train_free is false
        X = np.array(self.X)
        y = np.array(self.y)

        # find the number of classes in y there shoud be N + 1, where N is the number of objects in the scene and also the number of classifiers
        self.num_classifiers = len(list(np.unique(self.y))) - 1
        print(f"Number of classes: {self.num_classifiers}")

        # make a list to hold all of the classifiers
        self.clfs = []

        # loop through and build the classifiers
        for i in range(self.num_classifiers):
            # take a subset / do spatial filtering
            X = X[:,:,:] # Does nothing for now

            X_class = X[np.logical_or(y==0, y==(i+1)),:,:]
            y_class = y[np.logical_or(y==0, y==(i+1)),]

            # Try rebuilding the classifier each time
            if self.rebuild == True:
                self.next_fit_window = 0
                # tf.keras.backend.clear_session()

            subX = X_class[self.next_fit_window:,:,:]
            suby = y_class[self.next_fit_window:]
            self.next_fit_window = nwindows

            for train_idx, test_idx in self.cv.split(subX,suby):
                X_train, X_test = subX[train_idx], subX[test_idx]
                y_train, y_test = suby[train_idx], suby[test_idx]

                z_dim, y_dim, x_dim = X_train.shape
                X_train = X_train.reshape(z_dim, x_dim*y_dim)
                scaler_train = preprocessing.StandardScaler().fit(X_train)
                X_train_scaled = scaler_train.transform(X_train)

                print(f"The shape of X_train_scaled is {X_train_scaled.shape}")

                z_dim, y_dim, x_dim = X_test.shape
                X_test = X_test.reshape(z_dim, x_dim*y_dim)
                scaler_test = preprocessing.StandardScaler().fit(X_test)
                X_test_scaled = scaler_test.transform(X_test)

                if i == 0:
                    # Compile the model
                    print("\nWorking on first model...")
                    self.clf0and1.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    # Fit the model
                    self.clf0and1.fit(x=X_train_scaled, y=y_train, batch_size=5, epochs=4, shuffle=True, verbose=2, validation_data=(X_test_scaled, y_test)) # Need to reshape X_train
                    
                else:
                    print("\nWorking on second model...")
                    # Compile the model
                    self.clf0and2.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    # Fit the model
                    self.clf0and2.fit(x=X_train_scaled, y=y_train, batch_size=5, epochs=4, shuffle=True, verbose=2, validation_data=(X_test_scaled, y_test)) # Need to reshape X_train

            # Print performance stats
            # accuracy
            # correct = preds == self.y
            # #print(correct)

            self.offline_window_count = nwindows
            self.offline_window_counts.append(self.offline_window_count)
            # accuracy
            accuracy = sum(preds == self.y)/len(preds)
            self.offline_accuracy.append(accuracy)
            print("accuracy = {}".format(accuracy))
            # precision
            precision = precision_score(self.y,preds, average = 'micro')
            self.offline_precision.append(precision)
            print("precision = {}".format(precision))
            # recall
            recall = recall_score(self.y, preds, average = 'micro')
            self.offline_recall.append(recall)
            print("recall = {}".format(recall))
            # confusion matrix in command line
            cm = confusion_matrix(self.y, preds)
            self.offline_cm = cm
            print("confusion matrix")
            print(cm)

    def predict(self, X, print_predict):
        # if X is 2D, make it 3D with one as first dimension
        if len(X.shape) < 3:
            X = X[np.newaxis, ...]

        print("the shape of X is", X.shape)

        self.predict0and1 = Sequential([
            Flatten(),
            Dense(units=8, input_shape=(4,), activation='relu'),
            Dense(units=16, activation='relu'),
            Dense(units=3, activation='sigmoid')
        ])

        self.predict0and2 = Sequential([
            Flatten(),
            Dense(units=8, input_shape=(4,), activation='relu'),
            Dense(units=16, activation='relu'),
            Dense(units=3, activation='sigmoid')
        ])

        z_dim, y_dim, x_dim = X.shape
        X_predict = X.reshape(z_dim, x_dim*y_dim)
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

        return final_predictions
