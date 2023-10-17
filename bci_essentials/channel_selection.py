"""
This module includes functions for selecting channels in order to
improve BCI performance.

The EEG data input for each function is a set of windows. The data must
be of the shape `W x C x S`, where:
- W = number of windows
- C = number of channels
- S = number of samples

"""
from joblib import Parallel, delayed
import time
import numpy as np
import pandas as pd


def channel_selection_by_method(kernel_func, X, y, channel_labels,                                          # kernel setup
                                method = "SBS", metric="accuracy", initial_channels = [],                   # wrapper setup
                                max_time= 999, min_channels=1, max_channels=999, performance_delta= 0.001,  # stopping criterion
                                n_jobs=1, print_output="silent", record_performance=True):                  # njobs
    """
    Passes the BCI kernel function into a wrapper defined by method.

    Parameters
    ----------
    kernel_func : function
        The classification kernel function which does feature extraction
        and classification.
        Different functions  are used for MI, P300, SSVEP, etc.
    X : numpy.ndarray
        Training data for the classifier as windows of EEG data.
        3D array containing data with `float` type.

        shape = (`W_windows`,`C_channels`,`S_samples`)
    y : numpy.ndarray
        Training labels for the classifier.
        1D array.

    Returns:
    updated_subset, self.clf, preds, accuracy, precision, recall, record_performance, record_time

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
                n_jobs=n_jobs, print_output=print_output, record_performance=record_performance)

    elif method == "SFS":
        print("Initial subset: ", initial_channels)

        # pass arguments to SBS
        return sfs(kernel_func, X, y, channel_labels=channel_labels, 
                metric=metric, initial_channels=initial_channels, 
                max_time=max_time, min_channels=min_channels, max_channels=max_channels, performance_delta=performance_delta,
                n_jobs=n_jobs, print_output=print_output, record_performance=record_performance)

    elif method == "SBFS":
        if initial_channels == []:
            initial_channels = channel_labels

        print("Initial subset: ", initial_channels)

        # pass arguments to SBS
        return sbfs(kernel_func, X, y, channel_labels=channel_labels, 
                metric=metric, initial_channels=initial_channels, 
                max_time=max_time, min_channels=min_channels, max_channels=max_channels, performance_delta=performance_delta,
                n_jobs=n_jobs, print_output=print_output, record_performance=record_performance)

    elif method == "SFFS":
        print("Initial subset: ", initial_channels)

        # pass arguments to SBS
        return sffs(kernel_func, X, y, channel_labels=channel_labels, 
                metric=metric, initial_channels=initial_channels, 
                max_time=max_time, min_channels=min_channels, max_channels=max_channels, performance_delta=performance_delta,
                n_jobs=n_jobs, print_output=print_output, record_performance=record_performance)


def check_stopping_criterion(
    current_time,
    nchannels,
    current_performance_delta,
    max_time,
    min_channels,
    max_channels,
    performance_delta,
    print_output=True,
):
    """Function to check if a stopping criterion has been met.

    Parameters
    ----------
    current_time : float
        The time elapsed since the start of the channel selection method.
    nchannels : int
        The number of channels in the current iteration of the new best channel
        subset (`len(new_channel_subset)`).
    current_performance_delta : float
        The performance delta between the current iteration and the previous.
    max_time : int
        The maxiumum amount of time, in seconds, that the function will
        search for the optimal solution.
    min_channels : int
        The minimum number of channels.
    max_channels : int
        The maximum number of channels.
    performance_delta : float
        The performance delta under which the algorithm is considered to
        be close enough to optimal.
    print_output : str, *optional*
        Flag on whether or not to print output.
        - Default is `True`.

    Returns
    -------
    *bool*
        Has stopping criterion been met (`True`) or not (`False`).

    """
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
    
def sfs(kernel_func, X, y, channel_labels, 
        metric, initial_channels,
        max_time, min_channels, max_channels, performance_delta,
        n_jobs, print_output, record_performance):

    results_df = pd.DataFrame(columns=["Step", "Time", "N Channels", "Channel Subset", "Unique Combinations Tested in Step", "Accuracy", "Precision", "Recall"])
    step = 1


    start_time = time.time()

    nwindows, nchannels, nsamples = X.shape
    sfs_subset = []

    for i,c in enumerate(channel_labels):
        if c in initial_channels:
            sfs_subset.append(i)

    previous_performance = 0

    stop_criterion = False

    # Get the performance of the initial subset
    initial_model, initial_preds, initial_accuracy, initial_precision, initial_recall = kernel_func(X[:,sfs_subset,:],y)
    if metric == "accuracy":
        initial_performance = initial_accuracy
    elif metric == "precision":
        initial_performance = initial_precision
    elif metric == "recall":
        initial_performance = initial_recall

    # Best
    best_channel_subset = initial_channels
    best_model = initial_model
    best_performance = initial_performance
    best_preds = initial_preds
    best_accuracy = initial_accuracy
    best_precision = initial_precision
    best_recall = initial_recall

    preds = []
    accuracy = 0
    precision = 0
    recall = 0

    while(stop_criterion == False):
        sets_to_try = []
        X_to_try = []
        for c in range(nchannels):
            if c not in sfs_subset:
                set_to_try = sfs_subset.copy()
                set_to_try.append(c)
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
 
        sfs_subset = sets_to_try[best_set_index]
        new_channel_subset = [channel_labels[c] for c in sfs_subset]
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


        if current_performance > best_performance:
            best_channel_subset = new_channel_subset
            best_model = model
            best_performance = current_performance
            best_preds = preds
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
        elif current_performance >= best_performance and len(new_channel_subset) < len(best_channel_subset):
            best_channel_subset = new_channel_subset
            best_model = model
            best_performance = current_performance
            best_preds = preds
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall

        if record_performance == True:
            new_channel_subset.sort()
            results_df.loc[step] = [step, time.time()-start_time, len(new_channel_subset), "".join(new_channel_subset), len(sets_to_try), accuracy, precision, recall]

        step += 1

        stop_criterion = check_stopping_criterion(time.time() - start_time, len(new_channel_subset), p_delta, max_time, min_channels, max_channels, performance_delta, print_output=True)

    new_channel_subset = [channel_labels[c] for c in sfs_subset]

    if print_output == "verbose" or print_output == "final":
        print(new_channel_subset)
        print(metric, " : ", current_performance)
        print("Time to optimal subset: ", time.time()-start_time, "s")

    # Get the best model


    return best_channel_subset, best_model, best_preds, best_accuracy, best_precision, best_recall, results_df

def sbs(kernel_func, X, y, channel_labels, 
    metric, initial_channels,
    max_time, min_channels, max_channels, performance_delta,
    n_jobs, print_output, record_performance):

    results_df = pd.DataFrame(columns=["Step", "Time", "N Channels", "Channel Subset", "Unique Combinations Tested in Step", "Accuracy", "Precision", "Recall"])
    step = 1

    if len(initial_channels) <= min_channels:
        initial_channels = channel_labels

    start_time = time.time()

    nwindows, nchannels, nsamples = X.shape
    sbs_subset = []
    all_sets_tried = []             # set of all channels that have been tried

    for i,c in enumerate(channel_labels):
        if c in initial_channels:
            sbs_subset.append(i)

    performance_at_nchannels = np.zeros(len(sbs_subset))
    best_subset_at_nchannels = [0] * len(sbs_subset)

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
        for c in sbs_subset:
            set_to_try = sbs_subset.copy()
            set_to_try.remove(c)
            set_to_try.sort()

            # Only try sets that have not been tried before
            if set_to_try not in all_sets_tried:
                sets_to_try.append(set_to_try)
                all_sets_tried.append(set_to_try)
            else:
                continue

            # get the new X
            new_X = np.zeros((nwindows, len(set_to_try), nsamples))
            for i,j in enumerate(set_to_try):
                new_X[:,i,:] = X[:,j,:]

            X_to_try.append(new_X)

        # run the kernel function on all cores
        outputs = Parallel(n_jobs=n_jobs)(delayed(kernel_func)(Xtest,y) for Xtest in X_to_try) 

        # [all_sets_tried.append(set.sort()) for set in sets_to_try]
            
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
        sbs_subset = sets_to_try[best_set_index]
        new_channel_subset = [channel_labels[c] for c in sbs_subset]
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
        if performance_at_nchannels[len(sbs_subset)-1] < best_performance:
            performance_at_nchannels[len(sbs_subset)-1] = best_performance
            best_subset_at_nchannels[len(sbs_subset)-1] = sbs_subset

        p_delta = performance - previous_performance
        previous_performance = performance

        if record_performance == True:
            new_channel_subset.sort()
            results_df.loc[step] = [step, time.time()-start_time, len(new_channel_subset), "".join(new_channel_subset), len(sets_to_try), accuracy, precision, recall]

        step += 1

        stop_criterion = check_stopping_criterion(time.time() - start_time, len(new_channel_subset), p_delta, max_time, min_channels, max_channels, performance_delta, print_output=True)
    
    new_channel_subset = [channel_labels[c] for c in sbs_subset]

    if print_output == "verbose" or print_output == "final":
        print(new_channel_subset)
        print(metric, " : ", performance)
        print("Time to optimal subset: ", time.time()-start_time, "s")


    return new_channel_subset, model, preds, accuracy, precision, recall, results_df


def sbfs(kernel_func, X, y, channel_labels, 
    metric, initial_channels,
    max_time, min_channels, max_channels, performance_delta,
    n_jobs, print_output, record_performance):

    results_df = pd.DataFrame(columns=["Step", "Time", "N Channels", "Channel Subset", "Unique Combinations Tested in Step", "Accuracy", "Precision", "Recall"])
    step = 1

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
                all_sets_tried.append(set_to_try)
            else:
                continue

            # get the new X
            new_X = np.zeros((nwindows, len(set_to_try), nsamples))
            for i,j in enumerate(set_to_try):
                new_X[:,i,:] = X[:,j,:]

            X_to_try.append(new_X)

        # run the kernel function on all cores
        outputs = Parallel(n_jobs=n_jobs)(delayed(kernel_func)(Xtest,y) for Xtest in X_to_try) 

        # [all_sets_tried.append(set.sort()) for set in sets_to_try]
            
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
        if performance_at_nchannels[len(sbfs_subset)-1] < best_performance:
            performance_at_nchannels[len(sbfs_subset)-1] = best_performance
            best_subset_at_nchannels[len(sbfs_subset)-1] = sbfs_subset

        p_delta = performance - previous_performance
        previous_performance = performance

        if record_performance == True:
            new_channel_subset.sort()
            results_df.loc[step] = [step, time.time()-start_time, len(new_channel_subset), "".join(new_channel_subset), len(sets_to_try), accuracy, precision, recall]

        step += 1


        # Conditional Inclusion
        while(stop_criterion == False):
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
                    all_sets_tried.append(set_to_try)

                else:
                    continue

                # get the new X
                new_X = np.zeros((nwindows, len(set_to_try), nsamples))
                for i,j in enumerate(set_to_try):
                    new_X[:,i,:] = X[:,j,:]

                X_to_try.append(new_X)

            if X_to_try == []:
                break

            # run the kernel on the new sets
            outputs = Parallel(n_jobs=n_jobs)(delayed(kernel_func)(Xtest,y) for Xtest in X_to_try) 

            # [all_sets_tried.append(set.sort()) for set in sets_to_try]

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

                if record_performance == True:
                    new_channel_subset.sort()
                    results_df.loc[step] = [step, time.time()-start_time, len(new_channel_subset), "".join(new_channel_subset), len(sets_to_try), accuracy, precision, recall]
                    step += 1

                performance_at_nchannels[length_of_resultant_set-1] = performance
                best_subset_at_nchannels[length_of_resultant_set-1] = sbfs_subset

            # if no performance gains, then stop conditional inclusion
            else:
                break

            # Check stopping criterion
            stop_criterion = check_stopping_criterion(time.time() - start_time, len(new_channel_subset), p_delta, max_time, min_channels, max_channels, performance_delta, print_output=True)

        stop_criterion = check_stopping_criterion(time.time() - start_time, len(new_channel_subset), p_delta, max_time, min_channels, max_channels, performance_delta, print_output=True)
    
    new_channel_subset = [channel_labels[c] for c in sbfs_subset]

    if print_output == "verbose" or print_output == "final":
        print(new_channel_subset)
        print(metric, " : ", performance)
        print("Time to optimal subset: ", time.time()-start_time, "s")


    return new_channel_subset, model, preds, accuracy, precision, recall, results_df


def sffs(kernel_func, X, y, channel_labels, 
    metric, initial_channels,
    max_time, min_channels, max_channels, performance_delta,
    n_jobs, print_output, record_performance):

    results_df = pd.DataFrame(columns=["Step", "Time", "N Channels", "Channel Subset", "Unique Combinations Tested in Step", "Accuracy", "Precision", "Recall"])
    step = 1

    start_time = time.time()

    nwindows, nchannels, nsamples = X.shape
    sffs_subset = []
    all_sets_tried = []             # set of all channels that have been tried

    for i,c in enumerate(channel_labels):
        if c in initial_channels:
            sffs_subset.append(i)

    performance_at_nchannels = np.zeros(len(channel_labels))
    performance_at_nchannels[:min_channels-1] = np.inf
    best_subset_at_nchannels = [0] * len(channel_labels)

    previous_performance = 0

    stop_criterion = False

    preds = []
    accuracy = 0
    precision = 0
    recall = 0

    pass_stopping_criterion = False

    # TODO Test the initial subset


    while(stop_criterion == False):
        sets_to_try = []
        X_to_try = []
        for c in range(nchannels):
            if c not in sffs_subset:
                set_to_try = sffs_subset.copy()
                set_to_try.append(c)
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


        best_performance = np.max(performances)
        best_set_index = accuracies.index(best_performance)

        # else:
        sffs_subset = sets_to_try[best_set_index]
        new_channel_subset = [channel_labels[c] for c in sffs_subset]
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
        if performance_at_nchannels[len(sffs_subset)-1] < best_performance:
            performance_at_nchannels[len(sffs_subset)-1] = best_performance
            best_subset_at_nchannels[len(sffs_subset)-1] = sffs_subset

        p_delta = performance - previous_performance
        previous_performance = performance

        if record_performance == True:
            new_channel_subset.sort()
            results_df.loc[step] = [step, time.time()-start_time, len(new_channel_subset), "".join(new_channel_subset), len(sets_to_try), accuracy, precision, recall]

        step += 1


        # Conditional Exclusion
        while(stop_criterion == False):
            # Get the length of the set if we were to include an additional channel
            length_of_resultant_set = len(sffs_subset) - 1
            if length_of_resultant_set < min_channels or length_of_resultant_set == 0:
                break

            # If length of resultant set equal to min channels the pass stopping criterion
            if length_of_resultant_set == min_channels:
                pass_stopping_criterion = True

            # Check all of the possible inclusions that do not lead to a previously tested subset
            potential_channels_to_add = list(range(len(channel_labels)))
            [potential_channels_to_add.remove(c) for c in sffs_subset]

            sets_to_try = []
            X_to_try = []

            for c in sffs_subset:
                set_to_try = sffs_subset.copy()
                set_to_try.remove(c)
                set_to_try.sort()

                # Only try sets that have not been tried before
                if set_to_try not in all_sets_tried:
                    sets_to_try.append(set_to_try)
                    all_sets_tried.append(set_to_try)
                else:
                    continue

                # get the new X
                new_X = np.zeros((nwindows, len(set_to_try), nsamples))
                for i,j in enumerate(set_to_try):
                    new_X[:,i,:] = X[:,j,:]

                X_to_try.append(new_X)

            if X_to_try == []:
                break

            # run the kernel on the new sets
            outputs = Parallel(n_jobs=n_jobs)(delayed(kernel_func)(Xtest,y) for Xtest in X_to_try) 

            # [all_sets_tried.append(set.sort()) for set in sets_to_try]

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
                sffs_subset = sets_to_try[best_set_index]
                new_channel_subset = [channel_labels[c] for c in sffs_subset]
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

                if record_performance == True:
                    new_channel_subset.sort()
                    results_df.loc[step] = [step, time.time()-start_time, len(new_channel_subset), "".join(new_channel_subset), len(sets_to_try), accuracy, precision, recall]
                    step += 1

                performance_at_nchannels[length_of_resultant_set-1] = performance
                best_subset_at_nchannels[length_of_resultant_set-1] = sffs_subset

            # if no performance gains, then stop conditional exclusion
            else:
                break

            # Check stopping criterion
            if pass_stopping_criterion == False:
                stop_criterion = check_stopping_criterion(time.time() - start_time, len(new_channel_subset), p_delta, max_time, min_channels, max_channels, performance_delta, print_output=True)

        if pass_stopping_criterion == True:
            pass_stopping_criterion = False
            continue
        else:
            stop_criterion = check_stopping_criterion(time.time() - start_time, len(new_channel_subset), p_delta, max_time, min_channels, max_channels, performance_delta, print_output=True)
    
    new_channel_subset = [channel_labels[c] for c in sffs_subset]

    if print_output == "verbose" or print_output == "final":
        print(new_channel_subset)
        print(metric, " : ", performance)
        print("Time to optimal subset: ", time.time()-start_time, "s")


    return new_channel_subset, model, preds, accuracy, precision, recall, results_df