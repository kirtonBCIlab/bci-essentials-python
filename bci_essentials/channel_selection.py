"""
Channel Selection

This module includes functions for selecting channels in order to improve BCI performance.


Inputs:
 kernel_func    - a function, the classification kernel, which does feature extraction and classification, different for MI, P300, SSVEP, etc.
 X              - training data for the classifier (np array, dimensions are nwindows X nchannels X nsamples)
 y              - training labels for the classifier (np array, dimensions are nwindow X 1)
 channel_labels - the set of channel labels corresponding to nchannels 
 max_time       - the maximum amount of time, in seconds, that the function will search for the optimal solution
 metric         - the metric used to measure the "goodness" of the classifier, default is accuracy
 n_jobs         - number of threads to dedicate to this calculation

 Outputs:
 new_channel_subset     - the new best channel set
 model                  - the trained model
 preds                  - the predictions from the model, same shape as y
 accuracy               - classifier accuracy
 precision              - classifier precision
 recall                 - classifier recall


"""
from joblib import Parallel, delayed
import time
import numpy as np

def sbs(kernel_func, X, y, channel_labels, max_time= 999, metric="accuracy", n_jobs=1):
    nwindows, nchannels, nsamples = X.shape
    sbs_subset = list(range(nchannels))

    best_overall_accuracy = 0

    start_time = time.time()

    stop_criterion = False

    preds = []
    accuracy = 0
    precision = 0
    recall = 0

    while((time.time()-start_time) < max_time and len(sbs_subset) > 2 and stop_criterion == False):
        sets_to_try = []
        X_to_try = []
        for c in sbs_subset:
            set_to_try = sbs_subset.copy()
            set_to_try.remove(c)
            sets_to_try.append(set_to_try)

            # get the new X
            new_X = np.zeros((nwindows, len(set_to_try), nsamples))
            for i,j in enumerate(set_to_try):
                new_X[:,i,:] = X[:,j,:]


            X_to_try.append(new_X)

        # This handles the multithreading to check multiple channel combinations at once it n_jobs > 1
        outputs = Parallel(n_jobs=n_jobs)(delayed(kernel_func)(Xtest,y) for Xtest in X_to_try) 
            
        models = []
        predictions = []
        accuracies = []
        precisions = []
        recalls = []

        # Extract the outputs
        for output in outputs:
            models.append(output[0])
            predictions.append(output[1])
            accuracies.append(output[2])
            precisions.append(output[3])
            recalls.append(output[4])

        if metric == "accuracy":
            best_set_index = accuracies.index(np.max(accuracies))

        # If if starts getting worse
        if np.max(accuracies) < best_overall_accuracy:
            stop_criterion = True
            print("accuracy declined, stopping sbs")

        else:
            sbs_subset = sets_to_try[best_set_index]
            new_channel_subset = [channel_labels[c] for c in sbs_subset]
            model = models[best_set_index]
            preds = predictions[best_set_index]
            accuracy = accuracies[best_set_index]
            best_overall_accuracy = accuracy
            precision = precisions[best_set_index]
            recall = recalls[best_set_index]
            print("new subset ", new_channel_subset)
            print("accuracy ", accuracy)
            print("accuracies ", accuracies)

    new_channel_subset = [channel_labels[c] for c in sbs_subset]

    print("Time to optimal subset: ", time.time()-start_time, "s")

    return new_channel_subset, model, preds, accuracy, precision, recall

