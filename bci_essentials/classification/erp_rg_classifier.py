# Stock libraries
import os
import sys
import random
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

# Custom libraries
# - Append higher directory to import bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))
from classification.generic_classifier import Generic_classifier
from bci_essentials.visuals import *
from bci_essentials.signal_processing import *
from bci_essentials.channel_selection import *

class ERP_rg_classifier(Generic_classifier):
    def set_p300_clf_settings(self, 
                                n_splits = 3,                   # number of folds for cross-validation
                                lico_expansion_factor = 1,      # Linear Combination Oversampling expansion factor is the factor by which the number of ERPs in the training set will be expanded
                                oversample_ratio = 0,           # traditional oversampling, float from 0.1-1 resulting ratio of erp class to non-erp class, 0 for no oversampling
                                undersample_ratio = 0,          # traditional undersampling, float from 0.1-1 resulting ratio of erp class to non-erp classs, 0 for no undersampling 
                                random_seed = 42,               # random seed
                                covariance_estimator = 'scm'    # Covarianc estimator, see pyriemann Covariances
                                ):

        self.n_splits = n_splits                    
        self.lico_expansion_factor = lico_expansion_factor
        self.oversample_ratio = oversample_ratio
        self.undersample_ratio = undersample_ratio
        self.random_seed = random_seed
        self.covariance_estimator = covariance_estimator

    def add_to_train(self, decision_block, label_idx, print_training=True):
        if print_training:
            print("adding to training set")
        # n = number of channels
        # m = number of samples
        # p = number of epochs
        p,n,m = decision_block.shape

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

    def fit(self, n_splits = 2, plot_cm=False, plot_roc=False, lico_expansion_factor = 1, print_fit=True, print_performance=True):
        
        if print_fit:
            print("Fitting the model using RG")
            print(self.X.shape, self.y.shape)

        # Define the strategy for cross validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)

        # Define the classifier
        self.clf = make_pipeline(XdawnCovariances(estimator=self.covariance_estimator), TangentSpace(metric="riemann"), LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto'))

        # Init predictions to all false 
        preds = np.zeros(len(self.y))

        # 
        def erp_rg_kernel(X,y):
            for train_idx, test_idx in cv.split(X,y):
                y_train, y_test = y[train_idx], y[test_idx]

                X_train = X[train_idx]
                X_test = X[test_idx]

                #LICO
                if print_fit:
                    print ("Before LICO: Shape X",X_train.shape,"Shape y", y_train.shape)
                if sum(y_train) > 2:
                    if lico_expansion_factor > 1:
                        X_train, y_train = lico(X_train, y_train, expansion_factor=lico_expansion_factor, sum_num=2, shuffle=False)
                        if print_fit:
                            print("y_train =",y_train)
                if print_fit:
                    print("After LICO: Shape X",X_train.shape,"Shape y", y_train.shape)

                # Oversampling
                if self.oversample_ratio > 0:
                    p_count = sum(y_train)
                    n_count = len(y_train) - sum(y_train)

                    num_to_add = int(np.floor((self.oversample_ratio * n_count) - p_count))

                    # Add num_to_add random selections from the positive 
                    true_X_train = X_train[y_train == 1]

                    len_X_train = len(true_X_train)

                    for s in range(num_to_add):
                        to_add_X = true_X_train[random.randrange(0,len_X_train),:,:]

                        X_train = np.append(X_train,to_add_X[np.newaxis,:],axis=0)
                        y_train = np.append(y_train,[1],axis=0)
                    

                # Undersampling
                if self.undersample_ratio > 0:
                    p_count = sum(y_train)
                    n_count = len(y_train) - sum(y_train)

                    num_to_remove = int(np.floor(n_count - (p_count / self.undersample_ratio)))

                    ind_range = np.arange(len(y_train))
                    ind_list = list(ind_range)
                    to_remove = []

                    # Remove num_to_remove random selections from the negative
                    false_ind = list(ind_range[y_train == 0])

                    for s in range(num_to_remove):
                        # select a random value from the list of false indices
                        remove_at = false_ind[random.randrange(0,len(false_ind))]

                        # remove that value from the false ind list
                        false_ind.remove(remove_at)

                        # add the index to be removed to a list
                        to_remove.append(remove_at)

                    remaining_ind = ind_list
                    for i in range(len(to_remove)):
                        remaining_ind.remove(to_remove[i])

                    X_train = X_train[remaining_ind,:,:]
                    y_train = y_train[remaining_ind]


                self.clf.fit(X_train, y_train)
                preds[test_idx] = self.clf.predict(X_test)
                predproba = self.clf.predict_proba(X_test)

                # Use pred proba to show what would be predicted
                predprobs = predproba[:,1]
                real = np.where(y_test == 1)

                #TODO handle exception where two probabilities are the same
                prediction = int(np.where(predprobs == np.amax(predprobs))[0][0])

                if print_fit:
                    print("y_test =",y_test)

                    print(predproba)
                    print(real[0])
                    print(prediction)

            model = self.clf

            accuracy = sum(preds == self.y)/len(preds)
            precision = precision_score(self.y,preds)
            recall = recall_score(self.y, preds)

            return model, preds, accuracy, precision, recall

        
        # Check if channel selection is true
        if self.channel_selection_setup:
            print("Doing channel selection")
            # print("Initial subset ", self.chs_initial_subset)

            updated_subset, updated_model, preds, accuracy, precision, recall = channel_selection_by_method(erp_rg_kernel, self.X, self.y, self.channel_labels,             # kernel setup
                                                                            self.chs_method, self.chs_metric, self.chs_initial_subset,                                      # wrapper setup
                                                                            self.chs_max_time, self.chs_min_channels, self.chs_max_channels, self.chs_performance_delta,    # stopping criterion
                                                                            self.chs_n_jobs, self.chs_output)                                                               # njobs, output messages
                
            print("The optimal subset is ", updated_subset)

            self.subset = updated_subset
            self.clf = updated_model
        else:
            print("Not doing channel selection")
            self.clf, preds, accuracy, precision, recall = erp_rg_kernel(self.X, self.y)

        

        # Print performance stats
        # accuracy
        accuracy = sum(preds == self.y)/len(preds)
        self.offline_accuracy = accuracy
        if print_performance:
            print("accuracy = {}".format(accuracy))

        # precision
        precision = precision_score(self.y,preds)
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


        if plot_cm == True:
            cm = confusion_matrix(self.y, preds)
            ConfusionMatrixDisplay(cm).plot()
            plt.show()

        if plot_roc == True:
            print("plotting the ROC...")
            print("just kidding ROC has not been implemented")
