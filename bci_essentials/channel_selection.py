"""
Channel Selection

This module includes functions for selecting channels in order to improve BCI performance.

"""
from joblib import Parallel, delayed
import time
import numpy as np

def sbs(inner_func, X, y, channel_labels, max_time= 999, metric="accuracy", n_jobs=1):
    nwindows, nchannels, nsamples = X.shape
    sbs_subset = list(range(nchannels))

    best_overall_accuracy = 0

    start_time = time.time()

    stop_criterion = False

    preds = []
    accuarcy = 0
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

        outputs = Parallel(n_jobs=n_jobs)(delayed(inner_func)(Xtest,y) for Xtest in X_to_try) 
            
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

        # check



    return 

