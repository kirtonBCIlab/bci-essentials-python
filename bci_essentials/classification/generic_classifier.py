# Stock libraries
import numpy as np

class Generic_classifier():
    """ Parent classifier class """
    
    def __init__(self, training_selection=0, subset=[]):
        print("initializing the classifier")
        self.X = np.ndarray([0])
        self.y = np.ndarray([0])

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
        self.chs_output = print_output                  # output setting, silent, final, or verbose

        self.channel_selection_setup = True



    
    # add training data, to the training set using a decision block and a label
    def add_to_train(self, decision_block, labels, num_options = 0, meta = [], print_training=True):
        if print_training:
            print("adding to training set")
        # n = number of channels
        # m = number of samples
        # p = number of epochs
        p,n,m = decision_block.shape

        self.num_options = num_options
        self.meta = meta

        if self.X.size == 0:
            self.X = decision_block
            self.y = labels

        else:
            self.X = np.append(self.X, decision_block, axis=0)
            self.y = np.append(self.y, labels, axis=0)

    # predict a label based on a decision block
    def predict_decision_block(self, decision_block, print_predict=True):

        decision_block = self.get_subset(decision_block)


        if print_predict:
            print("making a prediction")

        # get prediction probabilities for all 
        proba_mat = self.clf.predict_proba(decision_block)

        proba = proba_mat[:,1]

        relative_proba = proba / np.amax(proba)

        log_proba = np.log(relative_proba)

        if print_predict:
            print("log relative probabilities")
            print(log_proba)

        # the selection is the highest probability

        prediction = int(np.where(proba == np.amax(proba))[0][0])

        self.predictions.append(prediction)
        self.pred_probas.append(proba_mat)

        return prediction
    
    def fit(self, **kwargs):
        """ Abstract method to fit classifier """
        return None

    def predict(self, **kwargs):
        """ Abstract method to predict with classifier"""
        return None