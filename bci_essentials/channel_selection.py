"""
Channel Selection

This module includes functions for selecting channels in order to improve BCI performance.


Inputs:
 kernel_func    - a function, the classification kernel, which does feature extraction and classification, different for MI, P300, SSVEP, etc.
 X              - training data for the classifier (np array, dimensions are nwindows X nchannels X nsamples)
 y              - training labels for the classifier (np array, dimensions are nwindow X 1)
 channel_labels - the set of channel labels corresponding to nchannels 
 max_time       - the maximum amount of time, in seconds, that the function will search for the optimal solution
 min_nchannels  - the minimum number of channels 
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


def channel_selection_by_method(kernel_func, X, y, channel_labels,                                          # kernel setup
                                method = "SBS", metric="accuracy", initial_channels = [],                   # wrapper setup
                                max_time= 999, min_channels=1, max_channels=999, performance_delta= 0.001,  # stopping criterion
                                n_jobs=1, print_output="silent"):                                                                  # njobs
    """
    Passes the BCI kernel function into a wrapper defined by method.

    kernel_func         - the kernel to be wrapped
    X                   - training data (nwindows X nchannels X nsamples)
    y                   - training labels (nwindows X 1)
    channel_labels      - channel labels, in a list of strings (nchannels X 1)

    method              - the wrapper method (ex. SBS, SFS, SFFS, SBFS)
    metric              - the method by which performance is measured, default is "accuracy"
    initial_channels    - initial guess of channels defaults to empty/full set for forward/backwardselections, respectively
    
    max_time            - max time for the algorithm to search
    min_channels        - min channels, default is 1
    max_channels        - max channels, default is nchannels
    performance_delta   - performance delta, under which the algorithm is considered to be close enough to optimal, default is 0.001

    Returns:
    updated_subset, self.clf, preds, accuracy, precision, recall

    """

    # max length can't be greater than the length of channel labels
    if max_channels > len(channel_labels):
        max_channels = len(channel_labels)

    if method == "SBS":
        if initial_channels == []:
            initial_channels = channel_labels

        print("Initial subset: ", initial_channels)

        # pass arguments to SBS
        return sbs(kernel_func, X, y, channel_labels=channel_labels, 
                metric=metric, initial_channels=initial_channels, 
                max_time=max_time, min_channels=min_channels, max_channels=max_channels, performance_delta=performance_delta,
                n_jobs=n_jobs, print_output=print_output)

    if method == "SBFS":
        if initial_channels == []:
            initial_channels = channel_labels

        print("Initial subset: ", initial_channels)

        # pass arguments to SBS
        return sbfs(kernel_func, X, y, channel_labels=channel_labels, 
                metric=metric, initial_channels=initial_channels, 
                max_time=max_time, min_channels=min_channels, max_channels=max_channels, performance_delta=performance_delta,
                n_jobs=n_jobs, print_output=print_output)

def check_stopping_criterion(current_time, nchannels, current_performance_delta, max_time, min_channels, max_channels, performance_delta, print_output=True):
    if current_time > max_time:
        if print_output == "verbose" or print_output == "final":
            print("Stopping based on time")
        return True

    elif nchannels <= min_channels:
        if print_output == "verbose" or print_output == "final":
            print("Stopping because minimum number of channels reached")
        return True

    elif nchannels >= max_channels:
        if print_output == "verbose" or print_output == "final":
            print("Stopping because maximum number of channels reached")
        return True

    elif current_performance_delta < performance_delta:
        if print_output == "verbose" or print_output == "final":
            print("Stopping because performance improvements are declining")
        return True

    else:
        return False

def sbs(kernel_func, X, y, channel_labels, 
        metric, initial_channels,
        max_time, min_channels, max_channels, performance_delta,
        n_jobs, print_output):

    start_time = time.time()

    nwindows, nchannels, nsamples = X.shape
    sbs_subset = []

    for i,c in enumerate(channel_labels):
        if c in initial_channels:
            sbs_subset.append(i)

    previous_performance = 0

    stop_criterion = False

    preds = []
    accuracy = 0
    precision = 0
    recall = 0

    while(stop_criterion == False):
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

            # make a list f all subsets of X to try
            X_to_try.append(new_X)

        # This handles the multiprocessing to check multiple channel combinations at once if n_jobs > 1
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

        # Get the performance metric
        if metric == "accuracy":
            performances = accuracies
        elif metric == "precision":
            performances = precisions
        elif metric == "recall":
            performances = recalls
        else:
            print("performance metric invalid, defaulting to accuracy")
            performances = accuracies

        # Get the index of the best X tried in this round
        best_set_index = accuracies.index(np.max(performances))

        # If it starts getting worse
        # if np.max(accuracies) < best_overall_accuracy:
        #     stop_criterion = True
        #     print("accuracy declined, stopping sbs")

        sbs_subset = sets_to_try[best_set_index]
        new_channel_subset = [channel_labels[c] for c in sbs_subset]
        model = models[best_set_index]
        preds = predictions[best_set_index]
        accuracy = accuracies[best_set_index]
        # best_overall_accuracy = accuracy
        precision = precisions[best_set_index]
        recall = recalls[best_set_index]
        if print_output =="verbose":
            print("new subset ", new_channel_subset)
            print("accuracy ", accuracy)
            print("accuracies ", accuracies)

        if metric == "accuracy":
            current_performance = accuracy

        p_delta = current_performance - previous_performance
        previous_performance = current_performance


        stop_criterion = check_stopping_criterion(time.time() - start_time, len(new_channel_subset), p_delta, max_time, min_channels, max_channels, performance_delta, print_output=True)

    new_channel_subset = [channel_labels[c] for c in sbs_subset]

    if print_output == "verbose" or print_output == "final":
        print(new_channel_subset)
        print(metric, " : ", current_performance)
        print("Time to optimal subset: ", time.time()-start_time, "s")

    return new_channel_subset, model, preds, accuracy, precision, recall

def sbfs(kernel_func, X, y, channel_labels, 
    metric, initial_channels,
    max_time, min_channels, max_channels, performance_delta,
    n_jobs, print_output):

    if len(initial_channels) <= min_channels:
        initial_channels = channel_labels

    start_time = time.time()

    nwindows, nchannels, nsamples = X.shape
    sbfs_subset = []
    all_sets_tried = []             # set of all channels that have been tried

    for i,c in enumerate(channel_labels):
        if c in initial_channels:
            sbfs_subset.append(i)

    performance_at_nchannels = np.zeros(len(sbfs_subset))
    best_subset_at_nchannels = [0] * len(sbfs_subset)

    previous_performance = 0

    stop_criterion = False

    preds = []
    accuracy = 0
    precision = 0
    recall = 0

    while(stop_criterion == False):
        # Exclusion Step
        sets_to_try = []
        X_to_try = []
        for c in sbfs_subset:
            set_to_try = sbfs_subset.copy()
            set_to_try.remove(c)
            set_to_try.sort()

            # Only try sets that have not been tried before
            if set_to_try not in all_sets_tried:
                sets_to_try.append(set_to_try)
            else:
                continue

            # get the new X
            new_X = np.zeros((nwindows, len(set_to_try), nsamples))
            for i,j in enumerate(set_to_try):
                new_X[:,i,:] = X[:,j,:]

            X_to_try.append(new_X)

        # run the kernel function on all cores
        outputs = Parallel(n_jobs=n_jobs)(delayed(kernel_func)(Xtest,y) for Xtest in X_to_try) 

        [all_sets_tried.append(set.sort()) for set in sets_to_try]
            
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

        # Get the performance metric
        if metric == "accuracy":
            performances = accuracies
        elif metric == "precision":
            performances = precisions
        elif metric == "recall":
            performances = recalls
        else:
            print("performance metric invalid, defaulting to accuracy")
            performances = accuracies


        best_performance = np.max(performances)
        best_set_index = accuracies.index(best_performance)

        # else:
        sbfs_subset = sets_to_try[best_set_index]
        new_channel_subset = [channel_labels[c] for c in sbfs_subset]
        model = models[best_set_index]
        preds = predictions[best_set_index]
        accuracy = accuracies[best_set_index]
        # best_overall_accuracy = accuracy
        precision = precisions[best_set_index]
        recall = recalls[best_set_index]

        performance = performances[best_set_index]
        if print_output == "verbose":
            print("Removed a channel")
            print("new subset ", new_channel_subset)
            print("accuracy ", accuracy)
            print("accuracies ", accuracies)

        # If this is the best perfomance at nchannels
        if performance_at_nchannels[len(sbfs_subset)] < best_performance:
            performance_at_nchannels[len(sbfs_subset)] = best_performance
            best_subset_at_nchannels[len(sbfs_subset)] = sbfs_subset

        p_delta = performance - previous_performance
        previous_performance = performance


        # Conditional Inclusion
        while(True):
            # Get the length of the set if we were to include an additional channel
            length_of_resultant_set = len(sbfs_subset) + 1
            if length_of_resultant_set > max_channels or length_of_resultant_set == len(channel_labels):
                break

            # Check all of the possible inclusions that do not lead to a previously tested subset
            potential_channels_to_add = list(range(len(channel_labels)))
            [potential_channels_to_add.remove(c) for c in sbfs_subset]

            sets_to_try = []
            X_to_try = []

            for c in potential_channels_to_add:
                set_to_try = sbfs_subset.copy()
                set_to_try.append(c)
                set_to_try.sort()

                if set_to_try not in all_sets_tried:
                    sets_to_try.append(set_to_try)

                else:
                    continue

                # get the new X
                new_X = np.zeros((nwindows, len(set_to_try), nsamples))
                for i,j in enumerate(set_to_try):
                    new_X[:,i,:] = X[:,j,:]

                X_to_try.append(new_X)

            # run the kernel on the new sets
            outputs = Parallel(n_jobs=n_jobs)(delayed(kernel_func)(Xtest,y) for Xtest in X_to_try) 

            [all_sets_tried.append(set.sort()) for set in sets_to_try]

            models = []
            predictions = []
            accuracies = []
            precisions = []
            recalls = []
            performances = []

            # Extract the outputs
            for output in outputs:
                models.append(output[0])
                predictions.append(output[1])
                accuracies.append(output[2])
                precisions.append(output[3])
                recalls.append(output[4])

            # Get the performance metric
            if metric == "accuracy":
                performances = accuracies
            elif metric == "precision":
                performances = precisions
            elif metric == "recall":
                performances = recalls
            else:
                print("performance metric invalid, defaulting to accuracy")
                performances = accuracies

            best_performance = np.max(performances)
            best_set_index = accuracies.index(best_performance)

            # if performance is better at that point
            if performance_at_nchannels[length_of_resultant_set-1] < best_performance:
                sbfs_subset = sets_to_try[best_set_index]
                new_channel_subset = [channel_labels[c] for c in sbfs_subset]
                model = models[best_set_index]
                preds = predictions[best_set_index]
                accuracy = accuracies[best_set_index]
                precision = precisions[best_set_index]
                recall = recalls[best_set_index]
                if print_output == "verbose":
                    print("Added back a channel")
                    print("new subset ", new_channel_subset)
                    print("accuracy ", accuracy)
                    print("accuracies ", accuracies)

                performance = best_performance

                p_delta = performance - previous_performance
                previous_performance = performance

                performance_at_nchannels[length_of_resultant_set-1] = performance
                best_subset_at_nchannels[length_of_resultant_set-1] = sbfs_subset

            # if no performance gains, then stop conditional inclusion
            else:
                break



            # If any of these are better than the best result for a given n channels then continue


            # Else continue with sbfs subset


        stop_criterion = check_stopping_criterion(time.time() - start_time, len(new_channel_subset), p_delta, max_time, min_channels, max_channels, performance_delta, print_output=True)
    
    new_channel_subset = [channel_labels[c] for c in sbfs_subset]

    if print_output == "verbose" or print_output == "final":
        print(new_channel_subset)
        print(metric, " : ", performance)
        print("Time to optimal subset: ", time.time()-start_time, "s")


    return new_channel_subset, model, preds, accuracy, precision, recall

    
    



