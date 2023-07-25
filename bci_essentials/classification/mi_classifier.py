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

# Custom libraries
# - Append higher directory to import bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))
from classification.generic_classifier import Generic_classifier
from bci_essentials.visuals import *
from bci_essentials.signal_processing import *
from bci_essentials.channel_selection import *

class MI_classifier(Generic_classifier):
    def set_mi_classifier_settings(self, n_splits=5, type="TS", remove_flats=False, whitening=False, covariance_estimator="scm", artifact_rejection="none", channel_selection="none", pred_threshold=0.5, random_seed = 42, n_jobs=1):
        # Build the cross-validation split
        self.n_splits = n_splits
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

        self.covariance_estimator = covariance_estimator
        
        # Shrinkage LDA
        if type == "sLDA":
            slda = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
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
            mdm = MDM(metric=dict(mean='riemann', distance='riemann'), n_jobs = n_jobs)
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
            # self.clf_model.steps.insert(0, ["Riemannian Potato", Potato()])
            # self.clf.steps.insert(0, ["Riemannian Potato", Potato()])

        if whitening == True:
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
        # get dimensions
        nwindows, nchannels, nsamples = self.X.shape 

        # do the rest of the training if train_free is false
        self.X = np.array(self.X)

        # Try rebuilding the classifier each time
        if self.rebuild == True:
            self.next_fit_window = 0
            self.clf = self.clf_model

        # get temporal subset
        subX = self.X[self.next_fit_window:,:,:]
        suby = self.y[self.next_fit_window:]
        self.next_fit_window = nwindows

        # Init predictions to all false 
        preds = np.zeros(nwindows)

        def mi_kernel(subX, suby):
            for train_idx, test_idx in self.cv.split(subX,suby):
                self.clf = self.clf_model

                X_train, X_test = subX[train_idx], subX[test_idx]
                y_train, y_test = suby[train_idx], suby[test_idx]

                # get the covariance matrices for the training set
                X_train_cov = Covariances(estimator=self.covariance_estimator).transform(X_train)
                X_test_cov = Covariances(estimator=self.covariance_estimator).transform(X_test)

                # fit the classsifier
                self.clf.fit(X_train_cov, y_train)
                preds[test_idx] = self.clf.predict(X_test_cov)

            accuracy = sum(preds == self.y)/len(preds)
            precision = precision_score(self.y,preds, average = 'micro')
            recall = recall_score(self.y, preds, average = 'micro')

            model = self.clf

            return model, preds, accuracy, precision, recall

        
        # Check if channel selection is true
        if self.channel_selection_setup:
            print("Doing channel selection")

            updated_subset, updated_model, preds, accuracy, precision, recall = channel_selection_by_method(mi_kernel, self.X, self.y, self.channel_labels,                      # kernel setup
                                                                            self.chs_method, self.chs_metric, self.chs_initial_subset,                                      # wrapper setup
                                                                            self.chs_max_time, self.chs_min_channels, self.chs_max_channels, self.chs_performance_delta,    # stopping criterion
                                                                            self.chs_n_jobs, self.chs_output)  
            # channel_selection_by_method(mi_kernel, subX, suby, self.channel_labels, method=self.chs_method, max_time=self.chs_max_time, metric="accuracy", n_jobs=-1)
                
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
        accuracy = sum(preds == self.y)/len(preds)
        self.offline_accuracy.append(accuracy)
        if print_performance:
            print("accuracy = {}".format(accuracy))

        # precision
        precision = precision_score(self.y, preds, average = 'micro')
        self.offline_precision.append(precision)
        if print_performance:
            print("precision = {}".format(precision))

        # recall
        recall = recall_score(self.y, preds, average = 'micro')
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
        # if X is 2D, make it 3D with one as first dimension
        if len(X.shape) < 3:
            X = X[np.newaxis, ...]

        X = self.get_subset(X)

        # Troubleshooting
        #X = self.X[-6:,:,:]

        if print_predict:
            print("the shape of X is", X.shape)

        X_cov = Covariances(estimator=self.covariance_estimator).transform(X)
        #X_cov = X_cov[0,:,:]

        pred = self.clf.predict(X_cov)
        pred_proba = self.clf.predict_proba(X_cov)

        if print_predict:
            print(pred)
            print(pred_proba)

        for i in range(len(pred)):
            self.predictions.append(pred[i])
            self.pred_probas.append(pred_proba[i])

        # add a threhold
        #pred = (pred_proba[:] >= self.pred_threshold).astype(int) # set threshold as 0.3
        #print(pred.shape)


        # print(pred)
        # for p in pred:
        #     p = int(p)
        #     print(p)
        # print(pred)

        # pred = str(pred).replace(".", ",")

        return pred
