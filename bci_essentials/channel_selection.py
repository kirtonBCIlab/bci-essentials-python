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


def channel_selection_by_method(
    kernel_func,
    X,
    y,
    channel_labels,
    method="SBS",
    metric="accuracy",
    initial_channels=[],
    max_time=999,
    min_channels=1,
    max_channels=999,
    performance_delta=0.001,
    n_jobs=1,
    print_output="silent",
):
    """Passes the BCI kernel function into a wrapper defined by `method`.

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

        shape = (`nwindows`)
    channel_labels : list of `str`
        The set of channel labels corresponding to `C_channels`.
        A list of strings with length = `C_channels`.
    method = str, *optional*
        The wrapper method. Options are `"SBS"` or `"SBFS"`.
        - Default is `"SBS"`.
    metric : str, *optional*
        The metric used to measure the "goodness" of the trained classifier.
        - Default is `"accuracy"`.
    initial_channels : list of `str`, *optional*
        Initial guess of channels.
        - Defaults is `[]`. Assigns an empty set for forward selections,
        and a full set for backward selections.
    max_time : int, *optional*
        The maxiumum amount of time, in seconds, that the function will
        search for the optimal solution.
        - Default is `999` seconds.
    min_channels : int, *optional*
        The minimum number of channels.
        - Default is `1`.
    max_channels : int, *optional*
        The maximum number of channels.
        - Default is `999`.
    performance_delta : float, *optional*
        The performance delta under which the algorithm is considered to
        be close enough to optimal.
        - Default is `0.001`.
    n_jobs : int, *optional*
        The number of threads to dedicate to this calculation.
        - Default is `1`.
    print_output : str, *optional*
        Flag on whether or not to print output. Options are:
        - `"verbose"`: Print output at each step.
        - `"final"`: Print output at the end.
        - `"silent"`: No output.
        - Default is `"silent"`.

    Returns
    -------
    new_channel_subset : list of `str`
        The new best channel subset from the list of `channel_labels`.
    self.clf : classifier
        The trained classification model.
    preds : numpy.ndarray
        The predictions from the model.
        1D array with the same shape as `y`.

        shape = (`nwindows`)
    accuracy : float
        The accuracy of the trained classification model.
    precision : float
        The precision of the trained classification model.
    recall : float
        The recall of the trained classification model.

    """

    # max length can't be greater than the length of channel labels
    if max_channels > len(channel_labels):
        max_channels = len(channel_labels)

    if method == "SBS":
        if initial_channels == []:
            initial_channels = channel_labels

        print("Initial subset: ", initial_channels)

        # pass arguments to SBS
        return sbs(
            kernel_func,
            X,
            y,
            channel_labels=channel_labels,
            metric=metric,
            initial_channels=initial_channels,
            max_time=max_time,
            min_channels=min_channels,
            max_channels=max_channels,
            performance_delta=performance_delta,
            n_jobs=n_jobs,
            print_output=print_output,
        )

    if method == "SBFS":
        if initial_channels == []:
            initial_channels = channel_labels

        print("Initial subset: ", initial_channels)

        # pass arguments to SBS
        return sbfs(
            kernel_func,
            X,
            y,
            channel_labels=channel_labels,
            metric=metric,
            initial_channels=initial_channels,
            max_time=max_time,
            min_channels=min_channels,
            max_channels=max_channels,
            performance_delta=performance_delta,
            n_jobs=n_jobs,
            print_output=print_output,
        )


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


def sbs(
    kernel_func,
    X,
    y,
    channel_labels,
    metric,
    initial_channels,
    max_time,
    min_channels,
    max_channels,
    performance_delta,
    n_jobs,
    print_output,
):
    """The SBS method for channel selection.

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

        shape = (`nwindows`)
    channel_labels: list of `str`
        The set of channel labels corresponding to `C_channels`.
        A list of strings with length = `C_channels`.
    metric : str
        The metric used to measure the "goodness" of the trained classifier.
    initial_channels : list of `str`
        Initial guess of channels.
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
    n_jobs : int
        The number of threads to dedicate to this calculation.
    print_output : str
        Flag on whether or not to print output. Options are:
        - `"verbose"`: Print output at each step.
        - `"final"`: Print output at the end.
        - `"silent"`: No output.

    Returns
    -------
    new_channel_subset : list of `str`
        The new best channel subset from the list of `channel_labels`.
    self.clf : classifier
        The trained classification model.
    preds : numpy.ndarray
        The predictions from the model.
        1D array with the same shape as `y`.

        shape = (`nwindows`)
    accuracy : float
        The accuracy of the trained classification model.
    precision : float
        The precision of the trained classification model.
    recall : float
        The recall of the trained classification model.


    """
    start_time = time.time()

    nwindows, nchannels, nsamples = X.shape
    sbs_subset = []

    for i, c in enumerate(channel_labels):
        if c in initial_channels:
            sbs_subset.append(i)

    previous_performance = 0

    stop_criterion = False

    preds = []
    accuracy = 0
    precision = 0
    recall = 0

    while stop_criterion is False:
        sets_to_try = []
        X_to_try = []
        for c in sbs_subset:
            set_to_try = sbs_subset.copy()
            set_to_try.remove(c)
            sets_to_try.append(set_to_try)

            # get the new X
            new_X = np.zeros((nwindows, len(set_to_try), nsamples))
            for i, j in enumerate(set_to_try):
                new_X[:, i, :] = X[:, j, :]

            # make a list f all subsets of X to try
            X_to_try.append(new_X)

        # This handles the multiprocessing to check multiple channel combinations at once if n_jobs > 1
        outputs = Parallel(n_jobs=n_jobs)(
            delayed(kernel_func)(Xtest, y) for Xtest in X_to_try
        )

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
        if print_output == "verbose":
            print("new subset ", new_channel_subset)
            print("accuracy ", accuracy)
            print("accuracies ", accuracies)

        if metric == "accuracy":
            current_performance = accuracy

        p_delta = current_performance - previous_performance
        previous_performance = current_performance

        stop_criterion = check_stopping_criterion(
            time.time() - start_time,
            len(new_channel_subset),
            p_delta,
            max_time,
            min_channels,
            max_channels,
            performance_delta,
            print_output=True,
        )

    new_channel_subset = [channel_labels[c] for c in sbs_subset]

    if print_output == "verbose" or print_output == "final":
        print(new_channel_subset)
        print(metric, " : ", current_performance)
        print("Time to optimal subset: ", time.time() - start_time, "s")

    return new_channel_subset, model, preds, accuracy, precision, recall


def sbfs(
    kernel_func,
    X,
    y,
    channel_labels,
    metric,
    initial_channels,
    max_time,
    min_channels,
    max_channels,
    performance_delta,
    n_jobs,
    print_output,
):
    """The SBFS method for channel selection.

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

        shape = (`nwindows`)
    channel_labels: list of `str`
        The set of channel labels corresponding to `C_channels`.
        A list of strings with length = `C_channels`.
    metric : str
        The metric used to measure the "goodness" of the trained classifier.
    initial_channels : list of `str`
        Initial guess of channels.
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
    n_jobs : int
        The number of threads to dedicate to this calculation.
    print_output : str
        Flag on whether or not to print output. Options are:
        - `"verbose"`: Print output at each step.
        - `"final"`: Print output at the end.
        - `"silent"`: No output.

    Returns
    -------
    new_channel_subset : list of `str`
        The new best channel subset from the list of `channel_labels`.
    self.clf : classifier
        The trained classification model.
    preds : numpy.ndarray
        The predictions from the model.
        1D array with the same shape as `y`.

        shape = (`nwindows`)
    accuracy : float
        The accuracy of the trained classification model.
    precision : float
        The precision of the trained classification model.
    recall : float
        The recall of the trained classification model.


    """
    if len(initial_channels) <= min_channels:
        initial_channels = channel_labels

    start_time = time.time()

    nwindows, nchannels, nsamples = X.shape
    sbfs_subset = []
    all_sets_tried = []  # set of all channels that have been tried

    for i, c in enumerate(channel_labels):
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

    while stop_criterion is False:
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
            for i, j in enumerate(set_to_try):
                new_X[:, i, :] = X[:, j, :]

            X_to_try.append(new_X)

        # run the kernel function on all cores
        outputs = Parallel(n_jobs=n_jobs)(
            delayed(kernel_func)(Xtest, y) for Xtest in X_to_try
        )

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
        while True:
            # Get the length of the set if we were to include an additional channel
            length_of_resultant_set = len(sbfs_subset) + 1
            if (
                length_of_resultant_set > max_channels
                or length_of_resultant_set == len(channel_labels)
            ):
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
                for i, j in enumerate(set_to_try):
                    new_X[:, i, :] = X[:, j, :]

                X_to_try.append(new_X)

            # run the kernel on the new sets
            outputs = Parallel(n_jobs=n_jobs)(
                delayed(kernel_func)(Xtest, y) for Xtest in X_to_try
            )

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
            if performance_at_nchannels[length_of_resultant_set - 1] < best_performance:
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

                performance_at_nchannels[length_of_resultant_set - 1] = performance
                best_subset_at_nchannels[length_of_resultant_set - 1] = sbfs_subset

            # if no performance gains, then stop conditional inclusion
            else:
                break

            # If any of these are better than the best result for a given n channels then continue

            # Else continue with sbfs subset

        stop_criterion = check_stopping_criterion(
            time.time() - start_time,
            len(new_channel_subset),
            p_delta,
            max_time,
            min_channels,
            max_channels,
            performance_delta,
            print_output=True,
        )

    new_channel_subset = [channel_labels[c] for c in sbfs_subset]

    if print_output == "verbose" or print_output == "final":
        print(new_channel_subset)
        print(metric, " : ", performance)
        print("Time to optimal subset: ", time.time() - start_time, "s")

    return new_channel_subset, model, preds, accuracy, precision, recall
