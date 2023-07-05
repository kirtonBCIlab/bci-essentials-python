"""
Classification Tools

"""


# Classification tools for transforming decision blocks into predictions

# Inputs are N x M x P where N = number of channels, M = number of samples, and P = number of signals / possible selections in P300

# Outputs a predction 

import string
import numpy as np
import random

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing

from pyriemann.preprocessing import Whitening
from pyriemann.estimation import ERPCovariances, XdawnCovariances, Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM, TSclassifier
#from pyriemann.utils.viz import plot_confusion_matrix
from pyriemann.channelselection import FlatChannelRemover, ElectrodeSelection
from pyriemann.clustering import Potato

import tensorflow as tf
from tensorflow import keras

from scipy import signal

#from mne.decoding import CSP

from bci_essentials.visuals import *
from bci_essentials.signal_processing import *
from bci_essentials.channel_selection import *

# TODO : move this to signal processing???
def lico(X,y,expansion_factor=3, sum_num=2, shuffle=False):

    """Oversampling (linear combination oversampling (LiCO))

    Samples random linear combinations of existing epochs of X.

    Parameters
    ----------
    X : numpy array 
        The file location of the spreadsheet
    y : numpy array
        A flag used to print the columns to the console
    expansion_factor : int, optional
        Number of times larger to make the output set over_X (default is 3)
    sum_num : int, optional
        Number of signals to be summed together (default is 2)

    Returns
    -------
    over_X : numpy array
        oversampled X
    over_y : numpy array
        oversampled y
    """

    true_X = X[y == 1]

    n,m,p = true_X.shape
    print("Shape of ERPs only ", true_X.shape)
    new_n = n*np.round(expansion_factor-1)
    new_X = np.zeros([new_n,m,p])
    for i in range(n):
        for j in range(sum_num):
            new_X[i,:,:] += true_X[random.choice(range(n)),:,:] / sum_num

    over_X = np.append(X,new_X,axis=0)
    over_y = np.append(y,np.ones([new_n]))

    return over_X, over_y

def get_ssvep_supertrial(X, 
        target_freqs, 
        fsample, 
        f_width=0.4, 
        n_harmonics=2, 
        covariance_estimator="scm"):
    """Get SSVEP Supertrial

    Creates the Riemannian Geometry supertrial for SSVEP

    Parameters
    ----------
    X : numpy array 
        Windows of EEG data, nwindows X nchannels X nsamples
    target_freqs : numpy array
        Target frequencies for the SSVEP
    fsample : float
        Sampling rate
    f_width : float, optional
        Width of frequency bins to be used around the target frequencies 
        (default 0.4)
    n_harmonics : int, optional
        Number of harmonics to be used for each frequency (default is 2)
    covarianc_estimator : str, optional
        Covariance Estimator (see Covariances - pyriemann) (default "scm")

    Returns
    -------
    super_X : numpy array
        Supertrials of X with the dimensions nwindows 
        by (nchannels*number of target_freqs) 
        by (nchannels*number of target_freqs)
    """
    nwindows, nchannels, nsamples = X.shape
    n_target_freqs = len(target_freqs)

    super_X = np.zeros([nwindows, nchannels*n_target_freqs, 
                        nchannels*n_target_freqs])

    # Create super trial of all trials filtered at all bands
    for w in range(nwindows):
        for tf, target_freq in enumerate(target_freqs):
            lower_bound = int((nchannels*tf))
            upper_bound = int((nchannels*tf)+nchannels)

            signal = X[w,:,:]
            for f in range(n_harmonics):
                if f == 0:
                    filt_signal = bandpass(signal, 
                                        f_low=target_freq-(f_width/2), 
                                        f_high=target_freq+(f_width/2), 
                                        order=5, 
                                        fsample=fsample)
                else:
                    filt_signal += bandpass(signal, 
                                        f_low=(target_freq*(f+1))-(f_width/2), 
                                        f_high=(target_freq*(f+1))+(f_width/2), 
                                        order=5, fsample=fsample)

            cov_mat = Covariances(estimator=covariance_estimator).transform(np.expand_dims(filt_signal, axis=0))

            cov_mat_diag = np.diag(np.diag(cov_mat[0,:,:]))

            super_X[w, lower_bound:upper_bound, lower_bound:upper_bound] = cov_mat_diag

    return super_X
    

# Write function that add to training set, fit, and predict

# make a generic classifier which can be extended to more specific classifiers
class generic_classifier():
    #
    def __init__(self, training_selection=0, subset=[]):
        print("initializing the classifier")
        self.X = []
        self.y = []

        #
        self.subset_defined = False
        self.subset = subset
        self.channel_labels = []
        self.channel_selection_setup = False

        # Lists for plotting classifier performance over time
        self.offline_accuracy = []
        self.offline_precision = []
        self.offline_recall = []
        self.offline_window_count = 0
        self.offline_window_counts = []

        # For iterative fitting,
        self.next_fit_window = 0

        # Keep track of predictions
        self.predictions = []
        self.pred_probas = []

    def get_subset(self, X=[]):
        """
        Get a subset of X according to labels or indices

        X               -   data in the shape of [# of windows, # of channels, # of samples]
        subset          -   list of indices (int) or labels (str) of the desired channels (default = [])
        channel_labels  -   channel labels from the entire EEG montage (default = [])
        """

        # Check for self.subset and/or self.channel_labels

        # Init
        subset_indices = []

        # Copy the indices based on subset
        try:
            # Check if we can use subset indices
            if self.subset == []:
                return X

            if type(self.subset[0]) == int:
                print("Using subset indices")

                subset_indices = self.subset

            # Or channel labels
            if type(self.subset[0]) == str:
                print("Using channel labels and subset labels")
                
                # Replace indices with those described by labels
                for sl in self.subset:
                    subset_indices.append(self.channel_labels.index(sl))

            # Return for the given indices
            try:
                # nwindows, nchannels, nsamples = self.X.shape

                if X == []:
                    new_X = self.X[:,subset_indices,:]
                    self.X = new_X
                else:
                    new_X = X[:,subset_indices,:]
                    X = new_X
                    return X


            except:
                # nchannels, nsamples = self.X.shape
                if X == []:
                    new_X = self.X[subset_indices,:]
                    self.X = new_X

                else:
                    new_X = X[subset_indices,:]
                    X = new_X
                    return X

        # notify if failed
        except:
            print("something went wrong, no subset taken")
            return X

    def setup_channel_selection(self, method = "SBS", metric="accuracy", initial_channels = [],             # wrapper setup
                                max_time= 999, min_channels=1, max_channels=999, performance_delta= 0.001,  # stopping criterion
                                n_jobs=1, print_output="silent"):                                                                  # njobs
        # Add these to settings later
        if initial_channels == []:
            self.chs_initial_subset = self.channel_labels
        else:
            self.chs_initial_subset = initial_channels
        self.chs_method = method                        # method to add/remove channels
        self.chs_metric = metric                        # metric by which to measure performance
        self.chs_n_jobs = n_jobs                        # number of threads
        self.chs_max_time = max_time                    # max time in seconds
        self.chs_min_channels = min_channels            # minimum number of channels
        self.chs_max_channels = max_channels            # maximum number of channels
        self.chs_performance_delta = performance_delta  # smallest performance increment to justify continuing search
        self.chs_output = print_output                        # output setting, silent, final, or verbose

        self.channel_selection_setup = True



    
    # add training data, to the training set using a decision block and a label
    def add_to_train(self, decision_block, labels, num_options = 0, meta = [], print_training=True):
        if print_training:
            print("adding to training set")
        # reshape from [n,m,p] to [p,n,m]
        # n = number of channels
        # m = number of samples
        # p = number of signals
        p,n,m = decision_block.shape
        # n,m,p = decision_block.shape


        # decision_block = self.get_subset(decision_block)

        self.num_options = num_options
        self.meta = meta

        # decision_block_reshape = np.swapaxes(np.swapaxes(decision_block,0,2),1,2)

        #print(labels)
            
        if self.X == []:
            self.X = decision_block
            self.y = labels

        else:
            # print(self.X.shape)
            # print(self.y.shape)
            self.X = np.append(self.X, decision_block, axis=0)
            self.y = np.append(self.y, labels, axis=0)

    # predict a label based on a decision block
    def predict_decision_block(self, decision_block, print_predict=True):

        decision_block = self.get_subset(decision_block)


        if print_predict:
            print("making a prediction")

        # # reshape from [n,m,p] to [p,n,m]
        # n,m,p = decision_block.shape
        #decision_block_reshape = np.swapaxes(np.swapaxes(decision_block,0,2),1,2)

        # # get prediction probabilities for all 
        # proba_mat = self.clf.predict_proba(decision_block_reshape)

        # decision_block = np.swapaxes(np.swapaxes(decision_block,0,2),1,2)

        # get prediction probabilities for all 
        proba_mat = self.clf.predict_proba(decision_block)

        proba = proba_mat[:,1]
        # print("probabilities:")
        # print(proba)

        relative_proba = proba / np.amax(proba)
        # print("relative probabiities")
        # print(relative_proba)

        log_proba = np.log(relative_proba)

        if print_predict:
            print("log relative probabilities")
            print(log_proba)

        # the selection is the highest probability

        prediction = int(np.where(proba == np.amax(proba))[0][0])

        self.predictions.append(prediction)
        self.pred_probas.append(proba_mat)

        return prediction


class erp_rg_classifier(generic_classifier):
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
        # p = number of windows
        p,n,m = decision_block.shape

        # get a subset
        decision_block = self.get_subset(decision_block)

        # get labels from label_idx
        labels = np.zeros([p])
        labels[label_idx] = 1
        if print_training:
            print(labels)

        # If the classifier has no data then initialize
        if self.X == []:
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
                        #to_remove.append(remove_at)

                        # add the index to be removed to a list
                        to_remove.append(remove_at)

                        #np.delete(false_ind,remove_at,axis=0)
                        
                        #ind_range

                        #np.delete(X_train,remove_at,axis=0)

                    #X_train = X_train[false]
                    #X_train = X_train[]

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

# SSVEP Classifier
class ssvep_riemannian_mdm_classifier(generic_classifier):
    """
    Classifies SSVEP based on relative band power at the expected frequencies
    """

    def set_ssvep_settings(self, n_splits=3, random_seed=42, n_harmonics=2, f_width=0.2, covariance_estimator="scm"):
        # Build the cross-validation split
        self.n_splits = n_splits
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

        self.rebuild = True

        self.n_harmonics = n_harmonics
        self.f_width = f_width
        self.covariance_estimator = covariance_estimator

        # Use an MDM classifier, maybe there will be other options later
        mdm = MDM(metric=dict(mean='riemann', distance='riemann'), n_jobs = 1)
        self.clf_model = Pipeline([("MDM", mdm)])
        self.clf = Pipeline([("MDM", mdm)])


    def fit(self, print_fit=True, print_performance=True):
        # get dimensions
        X = self.X


        # Convert each window of X into a SPD of dimensions [nwindows, nchannels*nfreqs, nchannels*nfreqs]
        nwindows, nchannels, nsamples = self.X.shape 

        #################
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

        def ssvep_kernel(subX, suby):
            for train_idx, test_idx in self.cv.split(subX,suby):
                self.clf = self.clf_model

                X_train, X_test = subX[train_idx], subX[test_idx]
                y_train, y_test = suby[train_idx], suby[test_idx]

                # get the covariance matrices for the training set
                X_train_super = get_ssvep_supertrial(X_train, self.target_freqs, fsample=256, n_harmonics=self.n_harmonics, f_width=self.f_width, covariance_estimator=self.covariance_estimator)
                X_test_super = get_ssvep_supertrial(X_test, self.target_freqs, fsample=256, n_harmonics=self.n_harmonics, f_width=self.f_width, covariance_estimator=self.covariance_estimator)

                # fit the classsifier
                self.clf.fit(X_train_super, y_train)
                preds[test_idx] = self.clf.predict(X_test_super)

            accuracy = sum(preds == self.y)/len(preds)
            precision = precision_score(self.y,preds, average="micro")
            recall = recall_score(self.y, preds, average="micro")

            model = self.clf

            return model, preds, accuracy, precision, recall

        # Check if channel selection is true
        if self.channel_selection_setup:
            print("Doing channel selection")

            updated_subset, updated_model, preds, accuracy, precision, recall = channel_selection_by_method(ssvep_kernel, self.X, self.y, self.channel_labels,             # kernel setup
                                                                            self.chs_method, self.chs_metric, self.chs_initial_subset,                                      # wrapper setup
                                                                            self.chs_max_time, self.chs_min_channels, self.chs_max_channels, self.chs_performance_delta,    # stopping criterion
                                                                            self.chs_n_jobs, self.chs_output) 
                
            print("The optimal subset is ", updated_subset)

            self.subset = updated_subset
            self.clf = updated_model
        else: 
            print("Not doing channel selection")
            self.clf, preds, accuracy, precision, recall= ssvep_kernel(subX, suby)

        # Print performance stats

        self.offline_window_count = nwindows
        self.offline_window_counts.append(self.offline_window_count)

        # accuracy
        accuracy = sum(preds == self.y)/len(preds)
        self.offline_accuracy.append(accuracy)
        if print_performance:
            print("accuracy = {}".format(accuracy))

        # precision
        precision = precision_score(self.y, preds, average='micro')
        self.offline_precision.append(precision)
        if print_performance:
            print("precision = {}".format(precision))

        # recall
        recall = recall_score(self.y, preds, average='micro')
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

        if print_predict:
            print("the shape of X is", X.shape)

        X_super = get_ssvep_supertrial(X, self.target_freqs, fsample=256, n_harmonics=self.n_harmonics, f_width=self.f_width)

        pred = self.clf.predict(X_super)
        pred_proba = self.clf.predict_proba(X_super)

        if print_predict:
            print(pred)
            print(pred_proba)

        for i in range(len(pred)):
            self.predictions.append(pred[i])
            self.pred_probas.append(pred_proba[i])

        return pred

# Train free classifier
# SSVEP CCA Classifier Sans Training
class ssvep_basic_classifier_tf(generic_classifier):
    """
    Classifies SSVEP based on relative bandpower, taking only the maximum
    """

    def set_ssvep_settings(self, sampling_freq, target_freqs):
        self.sampling_freq = sampling_freq
        self.target_freqs = target_freqs
        self.setup = False

    def fit(self, print_fit=True, print_performance=True):
        print("Oh deary me you must have mistaken me for another classifier which requires training")
        print("I DO NOT NEED TRAINING.")
        print("THIS IS MY FINAL FORM")
    

    def predict(self, X, print_predict):
        # get the shape
        nwindows, nchannels, nsamples = X.shape
        # The first time it is called it must be set up
        if self.setup == False:
            print("setting up the training free classifier")

            self.setup = True

        # Build one augmented channel, here by just adding them all together
        X = np.mean(X, axis=1)

        # Get the PSD estimate using Welch's method
        f, Pxx = signal.welch(X, fs=self.sampling_freq, nperseg=nsamples)
        
        # Get a vote for each window
        prediction = np.zeros(nwindows)
        for w in range(nwindows):
            # Get the frequency with the greatest PSD
            f_bins = np.zeros(len(self.target_freqs))
            Pxx_of_f_bins = np.zeros(len(self.target_freqs))
            for i, tf in enumerate(self.target_freqs):
                # Get the closest frequency bin
                f_bins[i] = np.argmin(np.abs(f - tf))

                Pxx_of_f_bins[i] = Pxx[w][int(f_bins[i])]
            
            prediction[w] = np.argmax(Pxx_of_f_bins)

        return prediction

# TODO : Add a SSVEP CCA Classifier

class mi_classifier(generic_classifier):
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


class switch_classifier_mdm(generic_classifier):
    def set_switch_classifier_mdm_settings(self, n_splits = 2, rebuild = True, random_seed = 42, activation_main = 'relu', activation_class = 'sigmoid'):

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
class switch_classifier_deep(generic_classifier):
    '''This is a switch_classifier. This means that classification occurs between neutral and one other label (i.e. Binary classification). 
    The produced probabilities between labels are then compared for one final classification.'''
    

    def set_switch_classifier_settings(self, n_splits = 2, rebuild = True, random_seed = 42, activation_main = 'relu', activation_class = 'sigmoid'):
        '''
        Function defines all basic settings for classification.

        Function has 6 parameters and defines two neural networks. One that will have weights and another that will not. 

        Parameters:
        self: Takes all variables from the class
        n_splits (int): Number of splits for StratifiedKFold
        rebuild (boolean): Resetting index for each call of fit
        random_seed (int): Ensures the same output for neural net each run if no parameters are changed
        activation_main (string): Activation function between hidden layers
        activation_class (string): Activation function for final layer in neural network

        Returns:
        Nothing - Networks created are used in fit()

        '''            
        # Definining activation functions
        self.activation_main = activation_main
        self.activation_class = activation_class

        # Defining training splits
        self.n_splits = n_splits
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        self.rebuild = rebuild

        self.random_seed = random_seed

        # Setting random seed for tensorflow so results remain the same for each model
        tf.random.set_seed(
            random_seed
        )

        '''Defining the neural network:
            self.clf: This is the classifier that will be trained and whose weights will differ. At the end of training the classifier is appended to a list
            self.clf_model: This will remain an unweighted version of the neural network and will be used to reset self.clf'''

        self.clf = Sequential([
            Flatten(),
            Dense(units=8, input_shape=(4,), activation=self.activation_main),
            Dense(units=16, activation=self.activation_main),
            Dense(units=3, activation=self.activation_class)
        ])

        self.clf_model = Sequential([
            Flatten(),
            Dense(units=8, input_shape=(4,), activation=self.activation_main),
            Dense(units=16, activation=self.activation_main),
            Dense(units=3, activation=self.activation_class)
        ])

    def fit(self, print_fit=True, print_performance=True):
        '''
        Fitting function for switch_classifier.

        Function uses the StratifiedKFold() function to split the data and then preprocess it using StandardScalar().
        The neural network is then fit and appended to a list before being reset.

        Parameters:
        self: Takes all variables from the class
        print_fit (boolean): Determines if we should print fitting info
        print_performance (boolean): Determines if we should print performance info

        Returns:
        Nothing - Models created used in predict()      
        '''
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
            X = X[:,:,:] # Does nothing for now

            # Changing the x array and y array so that their indicies match up and appropriate features are trained with appropraite labels
            # This is so training can be done on 0 vs 1 dataset and 0 vs 2 dataset
            X_class = X[np.logical_or(y==0, y==(i+1)),:,:]
            y_class = y[np.logical_or(y==0, y==(i+1)),]

            X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size=0.15, random_state=self.random_seed)

            # Try rebuilding the classifier each time
            if self.rebuild == True:
                self.next_fit_window = 0

            subX = X_class_train[self.next_fit_window:,:,:]
            suby = y_class_train[self.next_fit_window:]
            self.next_fit_window = nwindows

            preds = np.zeros((nwindows, self.num_classes))
            preds_multiclass = np.zeros(nwindows)

            for train_idx, test_idx in self.cv.split(subX,suby):
                X_train, X_test = subX[train_idx], subX[test_idx]
                y_train, y_test = suby[train_idx], suby[test_idx]

                # Reshaping the training data makes it easier to fit it to the neural network and other machine learning models
                z_dim, y_dim, x_dim = X_train.shape
                X_train = X_train.reshape(z_dim, x_dim*y_dim)
                # Scaling the data
                scaler_train = preprocessing.StandardScaler().fit(X_train)
                X_train_scaled = scaler_train.transform(X_train)

                # Repeating preprocessing steps done for training data on testing data
                z_dim, y_dim, x_dim = X_test.shape
                X_test = X_test.reshape(z_dim, x_dim*y_dim)
                scaler_test = preprocessing.StandardScaler().fit(X_test)
                X_test_scaled = scaler_test.transform(X_test)

                # Compile the model
                self.clf.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                # Fit the model
                self.clf.fit(x=X_train_scaled, y=y_train, batch_size=5, epochs=4, shuffle=True, verbose=2, validation_data=(X_test_scaled, y_test)) # Need to reshape X_train
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
                X_class_test = X_class_test.reshape(z_dim, x_dim*y_dim)
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
        '''
        Predict function which preprocessing data and makes prediction.

        Function is passed an array of size (X, 8, 512) from bci_data.py where it will predict upon the likelihood of state 1 vs state 2. Only works for three states currently.
        
        Parameters:
        self: Takes all class variables
        X (array): An array that will be predicted upon by previously trained models

        Returns:
        string_preds (string): Formatted predictions in the form of a string for Unity to process it
        '''

        # if X is 2D, make it 3D with one as first dimension
        if len(X.shape) < 3:
            X = X[np.newaxis, ...]

        print("the shape of X is", X.shape)

        # Reshaping data and preprocessing the same way as done in fit
        z_dim, y_dim, x_dim = X.shape
        X_predict = X.reshape(z_dim, x_dim*y_dim)
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
                temp_list.append(sub_list[iterations+1])

            iterations += 1
            final_preds.append(temp_list)
            temp_list = []

        '''This will format predictions so that unity can understand them. 
        However, it only works with two objects right now because of the x and y in zip'''
        
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

            final_string = ', '.join(string_list)

            print(f"final preds is: {final_preds}")
            print(f"string_preds are: {final_string}")
            
            return final_string
        except:
            print("Error - there are not an appropriate amount of labels (three) to complete predictions on")
            return None
                
class null_classifier(generic_classifier):
    def fit(self, print_fit=True, print_performance=True):

        print("This is a null classifier, there is no fitting")

    def predict(self, X, print_predict):
        return 0

