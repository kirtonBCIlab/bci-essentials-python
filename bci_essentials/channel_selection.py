"""
This module includes functions for selecting channels in order to
improve BCI performance.

The EEG data input for each function is a set of trials. The data must
be of the shape `n_trials x n_channels x n_samples`, where:
- n_trials = number of trials
- n_channels = number of channels
- n_samples = number of samples

"""

from joblib import Parallel, delayed
import time
import numpy as np
import pandas as pd
from .utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


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
    record_performance=True,
):
    """Passes the BCI kernel function into a wrapper defined by `method`.

    Parameters
    ----------
    kernel_func : function
        The classification kernel function which does feature extraction
        and classification.
        Different functions  are used for MI, P300, SSVEP, etc.
    X : numpy.ndarray
        Training data for the classifier as trials of EEG data.
        3D array containing data with `float` type.

        shape = (`n_trials`,`n_channels`,`n_samples`)
    y : numpy.ndarray
        Training labels for the classifier.
        1D array.

        shape = (`n_trials`)
    channel_labels : list of `str`
        The set of channel labels corresponding to `n_channels`.
        A list of strings with length = `n_channels`.
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
    record_performance : bool, *optional*
        Whether or not to record the performance of the channel selection
        - Default is `True`.

    Returns
    -------
    new_channel_subset : list of `str`
        The new best channel subset from the list of `channel_labels`.
    self.clf : classifier
        The trained classification model.
    preds : numpy.ndarray
        The predictions from the model.
        1D array with the same shape as `y`.

        shape = (`n_trials`)
    accuracy : float
        The accuracy of the trained classification model.
    precision : float
        The precision of the trained classification model.
    recall : float
        The recall of the trained classification model.
    results_df : pandas.DataFrame
        The dataframe containing the results of each step of channel selection.

    """

    # max length can't be greater than the length of channel labels
    if max_channels > len(channel_labels):
        logger.debug(
            "Maximum number of channels must be less than or equal to "
            + "the number of channels. Setting to number of channels."
        )
        max_channels = len(channel_labels)

    # min length can't be less than 1
    if min_channels < 1:
        logger.debug("Minimum number of channels must be greater than 0. Setting to 1.")
        min_channels = 1

    logger.debug("Running channel selection method: %s", method)
    if method == "SBS":
        if initial_channels == []:
            initial_channels = channel_labels
        logger.debug("Initial subset: %s", initial_channels)

        # pass arguments to SBS
        return __sbs(
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
            record_performance=record_performance,
        )

    elif method == "SFS":
        logger.debug("Initial subset: %s", initial_channels)

        # pass arguments to SBS
        return __sfs(
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
            record_performance=record_performance,
        )

    elif method == "SBFS":
        if initial_channels == []:
            initial_channels = channel_labels
        logger.debug("Initial subset: %s", initial_channels)

        # pass arguments to SBS
        return __sbfs(
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
            record_performance=record_performance,
        )

    elif method == "SFFS":
        logger.debug("Initial subset: %s", initial_channels)

        # pass arguments to SBS
        return __sffs(
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
            record_performance=record_performance,
        )


def __check_stopping_criterion(
    current_time,
    n_channels,
    current_performance_delta,
    max_time,
    min_channels,
    max_channels,
    performance_delta,
):
    """Function to check if a stopping criterion has been met.

    Parameters
    ----------
    current_time : float
        The time elapsed since the start of the channel selection method.
    n_channels : int
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

    Returns
    -------
    *bool*
        Has stopping criterion been met (`True`) or not (`False`).

    """
    if current_time > max_time:
        logger.debug("Stopping based on time")
        return True

    elif n_channels <= min_channels:
        logger.debug("Stopping because minimum number of channels reached")
        return True

    elif n_channels >= max_channels:
        logger.debug("Stopping because maximum number of channels reached")
        return True

    elif current_performance_delta < performance_delta:
        logger.debug("Stopping because performance improvements are declining")
        return True
    else:
        return False


def __sfs(
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
    record_performance,
):
    """
    The Sequential Forward Selection (SFS) method for channel selection.

    Parameters
    ----------
    kernel_func : function
        The classification kernel function which does feature extraction
        and classification.
        Different functions  are used for MI, P300, SSVEP, etc.
    X : numpy.ndarray
        Training data for the classifier as trials of EEG data.
        3D array containing data with `float` type.

        shape = (`n_trials`,`n_channels`,`n_samples`)
    y : numpy.ndarray
        Training labels for the classifier.
        1D array.

        shape = (`n_trials`)
    channel_labels: list of `str`
        The set of channel labels corresponding to `n_channels`.
        A list of strings with length = `n_channels`.
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
    record_performance : bool
        Flag on whether or not to record performance at each step.


    Returns
    -------
    new_channel_subset : list of `str`
        The new best channel subset from the list of `channel_labels`.
    self.clf : classifier
        The trained classification model.
    preds : numpy.ndarray
        The predictions from the model.
        1D array with the same shape as `y`.

        shape = (`n_trials`)
    accuracy : float
        The accuracy of the trained classification model.
    precision : float
        The precision of the trained classification model.
    recall : float
        The recall of the trained classification model.
    results_df : pandas.DataFrame
        The dataframe containing the results of each step of channel selection.
    """
    results_df = pd.DataFrame(
        columns=[
            "Step",
            "Time",
            "N Channels",
            "Channel Subset",
            "Unique Combinations Tested in Step",
            "Accuracy",
            "Precision",
            "Recall",
        ]
    )
    step = 1

    start_time = time.time()

    n_trials, n_channels, n_samples = X.shape
    sfs_subset = []

    for i, c in enumerate(channel_labels):
        if c in initial_channels:
            sfs_subset.append(i)

    previous_performance = 0

    stop_criterion = False

    # Get the performance of the initial subset, if possible
    try:
        (
            initial_model,
            initial_preds,
            initial_accuracy,
            initial_precision,
            initial_recall,
        ) = kernel_func(X[:, sfs_subset, :], y)
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

    # If not possible then set the initial performance to 0
    except ValueError:
        best_channel_subset = []
        best_model = None
        best_performance = 0
        best_preds = []
        best_accuracy = 0
        best_precision = 0
        best_recall = 0

    preds = []
    accuracy = 0
    precision = 0
    recall = 0

    while stop_criterion is False:
        sets_to_try = []
        X_to_try = []
        for channel in range(n_channels):
            if channel not in sfs_subset:
                set_to_try = sfs_subset.copy()
                set_to_try.append(c)
                sets_to_try.append(set_to_try)

                # Get the new subset of data
                new_subset_data = np.zeros((n_trials, len(set_to_try), n_samples))
                for subset_idx, channel_number in enumerate(set_to_try):
                    channel_data = X[:, channel_number, :]
                    new_subset_data[:, subset_idx, :] = channel_data

                # Add to list of all subsets of X to try
                X_to_try.append(new_subset_data)

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
            logger.warning("Performance metric invalid, defaulting to accuracy")
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
        logger.debug("New subset: %s", new_channel_subset)
        logger.debug("Accuracy: %s", accuracy)
        logger.debug("Accuracies: %s", accuracies)

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
        elif current_performance >= best_performance and len(new_channel_subset) < len(
            best_channel_subset
        ):
            best_channel_subset = new_channel_subset
            best_model = model
            best_performance = current_performance
            best_preds = preds
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall

        if record_performance is True:
            new_channel_subset.sort()
            results_df.loc[step] = [
                step,
                time.time() - start_time,
                len(new_channel_subset),
                "".join(new_channel_subset),
                len(sets_to_try),
                accuracy,
                precision,
                recall,
            ]

        step += 1

        stop_criterion = __check_stopping_criterion(
            time.time() - start_time,
            len(new_channel_subset),
            p_delta,
            max_time,
            min_channels,
            max_channels,
            performance_delta,
        )

    new_channel_subset = [channel_labels[c] for c in sfs_subset]

    logger.debug("Best channel subset: %s", best_channel_subset)
    logger.debug("%s : %s", metric, best_performance)
    logger.debug("Time to optimal subset: %s s", time.time() - start_time)

    # Get the best model

    return (
        best_channel_subset,
        best_model,
        best_preds,
        best_accuracy,
        best_precision,
        best_recall,
        results_df,
    )


def __sbs(
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
    record_performance,
):
    """The Sequential Backward Selection (SBS) method for channel selection.

    Parameters
    ----------
    kernel_func : function
        The classification kernel function which does feature extraction
        and classification.
        Different functions  are used for MI, P300, SSVEP, etc.
    X : numpy.ndarray
        Training data for the classifier as trials of EEG data.
        3D array containing data with `float` type.

        shape = (`n_trials`,`n_channels`,`n_samples`)
    y : numpy.ndarray
        Training labels for the classifier.
        1D array.

        shape = (`n_trials`)
    channel_labels: list of `str`
        The set of channel labels corresponding to `n_channels`.
        A list of strings with length = `n_channels`.
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
    record_performance : bool
        Flag on whether or not to record performance metrics at each step.

    Returns
    -------
    new_channel_subset : list of `str`
        The new best channel subset from the list of `channel_labels`.
    self.clf : classifier
        The trained classification model.
    preds : numpy.ndarray
        The predictions from the model.
        1D array with the same shape as `y`.

        shape = (`n_trials`)
    accuracy : float
        The accuracy of the trained classification model.
    precision : float
        The precision of the trained classification model.
    recall : float
        The recall of the trained classification model.
    results_df : pandas.DataFrame
        A dataframe containing the performance metrics at each step.


    """

    results_df = pd.DataFrame(
        columns=[
            "Step",
            "Time",
            "N Channels",
            "Channel Subset",
            "Unique Combinations Tested in Step",
            "Accuracy",
            "Precision",
            "Recall",
        ]
    )
    step = 1

    if len(initial_channels) <= min_channels:
        initial_channels = channel_labels

    start_time = time.time()

    n_trials, n_channels, n_samples = X.shape
    sbs_subset = []
    all_sets_tried = []  # set of all channels that have been tried

    for i, c in enumerate(channel_labels):
        if c in initial_channels:
            sbs_subset.append(i)

    # Get the performance of the initial subset
    (
        initial_model,
        initial_preds,
        initial_accuracy,
        initial_precision,
        initial_recall,
    ) = kernel_func(X[:, sbs_subset, :], y)
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

            # Get the new subset of data
            new_subset_data = np.zeros((n_trials, len(set_to_try), n_samples))
            for subset_idx, channel_number in enumerate(set_to_try):
                channel_data = X[:, channel_number, :]
                new_subset_data[:, subset_idx, :] = channel_data

            # Add to list of all subsets of X to try
            X_to_try.append(new_subset_data)

        # run the kernel function on all cores
        outputs = Parallel(n_jobs=n_jobs)(
            delayed(kernel_func)(Xtest, y) for Xtest in X_to_try
        )

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
            logger.warning("Performance metric invalid, defaulting to accuracy")
            performances = accuracies

        best_set_index = accuracies.index(np.max(performances))

        sbs_subset = sets_to_try[best_set_index]
        new_channel_subset = [channel_labels[c] for c in sbs_subset]
        model = models[best_set_index]
        preds = predictions[best_set_index]
        accuracy = accuracies[best_set_index]
        # best_overall_accuracy = accuracy
        precision = precisions[best_set_index]
        recall = recalls[best_set_index]

        current_performance = performances[best_set_index]
        logger.debug("Removed a channel")
        logger.debug("New subset: %s", new_channel_subset)
        logger.debug("Accuracy: %s", accuracy)
        logger.debug("Accuracies: %s", accuracies)

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
        elif current_performance >= best_performance and len(new_channel_subset) < len(
            best_channel_subset
        ):
            best_channel_subset = new_channel_subset
            best_model = model
            best_performance = current_performance
            best_preds = preds
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall

        if record_performance is True:
            new_channel_subset.sort()
            results_df.loc[step] = [
                step,
                time.time() - start_time,
                len(new_channel_subset),
                "".join(new_channel_subset),
                len(sets_to_try),
                accuracy,
                precision,
                recall,
            ]

        step += 1

        # Break if SBFS subset is 1 channel
        if len(sbs_subset) == 1:
            break

        stop_criterion = __check_stopping_criterion(
            time.time() - start_time,
            len(new_channel_subset),
            p_delta,
            max_time,
            min_channels,
            max_channels,
            performance_delta,
        )

    new_channel_subset = [channel_labels[c] for c in sbs_subset]

    logger.debug("Best channel subset: %s", best_channel_subset)
    logger.debug("%s : %s", metric, best_performance)
    logger.debug("Time to optimal subset: %s s", time.time() - start_time)

    return (
        best_channel_subset,
        best_model,
        best_preds,
        best_accuracy,
        best_precision,
        best_recall,
        results_df,
    )


def __sbfs(
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
    record_performance,
):
    """The Sequential Backward Floating Selection (SBFS) method for channel selection.

    Parameters
    ----------
    kernel_func : function
        The classification kernel function which does feature extraction
        and classification.
        Different functions  are used for MI, P300, SSVEP, etc.
    X : numpy.ndarray
        Training data for the classifier as trials of EEG data.
        3D array containing data with `float` type.

        shape = (`n_trials`,`n_channels`,`n_samples`)
    y : numpy.ndarray
        Training labels for the classifier.
        1D array.

        shape = (`n_trials`)
    channel_labels: list of `str`
        The set of channel labels corresponding to `n_channels`.
        A list of strings with length = `n_channels`.
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
    record_performance : bool
        Flag on whether or not to record performance metrics at each step.

    Returns
    -------
    new_channel_subset : list of `str`
        The new best channel subset from the list of `channel_labels`.
    self.clf : classifier
        The trained classification model.
    preds : numpy.ndarray
        The predictions from the model.
        1D array with the same shape as `y`.

        shape = (`n_trials`)
    accuracy : float
        The accuracy of the trained classification model.
    precision : float
        The precision of the trained classification model.
    recall : float
        The recall of the trained classification model.
    results_df : pandas.DataFrame
        A dataframe containing the performance metrics at each step.


    """
    results_df = pd.DataFrame(
        columns=[
            "Step",
            "Time",
            "N Channels",
            "Channel Subset",
            "Unique Combinations Tested in Step",
            "Accuracy",
            "Precision",
            "Recall",
        ]
    )
    step = 1

    if len(initial_channels) <= min_channels or len(initial_channels) == 0:
        initial_channels = channel_labels

    start_time = time.time()

    n_trials, n_channels, n_samples = X.shape
    sbfs_subset = []
    all_sets_tried = []  # set of all channels that have been tried

    for i, c in enumerate(channel_labels):
        if c in initial_channels:
            sbfs_subset.append(i)

    performance_at_nchannels = np.zeros(len(channel_labels))
    best_subset_at_nchannels = [0] * len(channel_labels)

    previous_performance = 0

    stop_criterion = False

    # Get the performance of the initial subset, if possible
    try:
        (
            initial_model,
            initial_preds,
            initial_accuracy,
            initial_precision,
            initial_recall,
        ) = kernel_func(X[:, sbfs_subset, :], y)
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

        performance_at_nchannels[len(initial_channels) - 1] = initial_performance
        best_subset_at_nchannels[len(initial_channels) - 1] = initial_channels

    # If not possible then set the initial performance to 0
    except ValueError:
        best_channel_subset = []
        best_model = None
        best_performance = 0
        best_preds = []
        best_accuracy = 0
        best_precision = 0
        best_recall = 0

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
                all_sets_tried.append(set_to_try)
            else:
                continue

            # Get the new subset of data
            new_subset_data = np.zeros((n_trials, len(set_to_try), n_samples))
            for subset_idx, channel_number in enumerate(set_to_try):
                channel_data = X[:, channel_number, :]
                new_subset_data[:, subset_idx, :] = channel_data

            # Add to list of all subsets of X to try
            X_to_try.append(new_subset_data)

        # run the kernel function on all cores
        outputs = Parallel(n_jobs=n_jobs)(
            delayed(kernel_func)(Xtest, y) for Xtest in X_to_try
        )

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
            logger.warning("Performance metric invalid, defaulting to accuracy")
            performances = accuracies

        best_round_performance = np.max(performances)
        best_set_index = accuracies.index(best_round_performance)

        sbfs_subset = sets_to_try[best_set_index]
        new_channel_subset = [channel_labels[c] for c in sbfs_subset]
        model = models[best_set_index]
        preds = predictions[best_set_index]
        accuracy = accuracies[best_set_index]
        precision = precisions[best_set_index]
        recall = recalls[best_set_index]

        logger.debug("Removed a channel")
        logger.debug("New subset: %s", new_channel_subset)
        logger.debug("Accuracy: %s", accuracy)
        logger.debug("Accuracies: %s", accuracies)

        current_performance = best_round_performance

        # If this is the best perfomance at n_channels
        if performance_at_nchannels[len(sbfs_subset) - 1] < current_performance:
            performance_at_nchannels[len(sbfs_subset) - 1] = current_performance
            best_subset_at_nchannels[len(sbfs_subset) - 1] = sbfs_subset

        p_delta = current_performance - previous_performance
        previous_performance = current_performance

        # If the performance is the best so far, then save it as the best
        if current_performance > best_performance:
            best_channel_subset = new_channel_subset
            best_model = model
            best_performance = current_performance
            best_preds = preds
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
        elif current_performance >= best_performance and len(new_channel_subset) < len(
            best_channel_subset
        ):
            best_channel_subset = new_channel_subset
            best_model = model
            best_performance = current_performance
            best_preds = preds
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall

        if record_performance:
            new_channel_subset.sort()
            results_df.loc[step] = [
                step,
                time.time() - start_time,
                len(new_channel_subset),
                "".join(new_channel_subset),
                len(sets_to_try),
                accuracy,
                precision,
                recall,
            ]

        step += 1

        # Conditional Inclusion
        while stop_criterion is False:
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
                    all_sets_tried.append(set_to_try)

                else:
                    continue

                # Get the new subset of data
                new_subset_data = np.zeros((n_trials, len(set_to_try), n_samples))
                for subset_idx, channel_number in enumerate(set_to_try):
                    channel_data = X[:, channel_number, :]
                    new_subset_data[:, subset_idx, :] = channel_data

                # Add to list of all subsets of X to try
                X_to_try.append(new_subset_data)

            if X_to_try == []:
                break

            # run the kernel on the new sets
            outputs = Parallel(n_jobs=n_jobs)(
                delayed(kernel_func)(Xtest, y) for Xtest in X_to_try
            )

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
                logger.warning("Performance metric invalid, defaulting to accuracy")
                performances = accuracies

            best_round_performance = np.max(performances)
            best_set_index = accuracies.index(best_round_performance)

            # if performance is better the best performance at n_channels
            if (
                performance_at_nchannels[length_of_resultant_set - 1]
                < best_round_performance
            ):
                sbfs_subset = sets_to_try[best_set_index]
                new_channel_subset = [channel_labels[c] for c in sbfs_subset]
                model = models[best_set_index]
                preds = predictions[best_set_index]
                accuracy = accuracies[best_set_index]
                precision = precisions[best_set_index]
                recall = recalls[best_set_index]

                logger.debug("Added back a channel")
                logger.debug("New subset: %s", new_channel_subset)
                logger.debug("Accuracy: %s", accuracy)
                logger.debug("Accuracies: %s", accuracies)

                current_performance = best_round_performance

                p_delta = current_performance - previous_performance
                previous_performance = current_performance

                # ADD Memory here
                if current_performance > best_performance:
                    best_channel_subset = new_channel_subset
                    best_model = model
                    best_performance = current_performance
                    best_preds = preds
                    best_accuracy = accuracy
                    best_precision = precision
                    best_recall = recall
                elif current_performance >= best_performance and len(
                    new_channel_subset
                ) < len(best_channel_subset):
                    best_channel_subset = new_channel_subset
                    best_model = model
                    best_performance = current_performance
                    best_preds = preds
                    best_accuracy = accuracy
                    best_precision = precision
                    best_recall = recall

                if record_performance:
                    new_channel_subset.sort()
                    results_df.loc[step] = [
                        step,
                        time.time() - start_time,
                        len(new_channel_subset),
                        "".join(new_channel_subset),
                        len(sets_to_try),
                        accuracy,
                        precision,
                        recall,
                    ]
                    step += 1

                performance_at_nchannels[length_of_resultant_set - 1] = (
                    current_performance
                )
                best_subset_at_nchannels[length_of_resultant_set - 1] = sbfs_subset

            # if no performance gains, then stop conditional inclusion
            else:
                break

            # Check stopping criterion
            stop_criterion = __check_stopping_criterion(
                time.time() - start_time,
                len(new_channel_subset),
                p_delta,
                max_time,
                min_channels,
                max_channels,
                performance_delta,
            )

        stop_criterion = __check_stopping_criterion(
            time.time() - start_time,
            len(new_channel_subset),
            p_delta,
            max_time,
            min_channels,
            max_channels,
            performance_delta,
        )

        # Break if SBFS subset is 1 channel
        if len(sbfs_subset) == 1:
            break

    new_channel_subset = [channel_labels[c] for c in sbfs_subset]

    logger.debug("Best channel subset: %s", best_channel_subset)
    logger.debug("%s : %s", metric, best_performance)
    logger.debug("Time to optimal subset: %s s", time.time() - start_time)

    return (
        best_channel_subset,
        best_model,
        best_preds,
        best_accuracy,
        best_precision,
        best_recall,
        results_df,
    )


def __sffs(
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
    record_performance,
):
    """The Sequential Forward Floating Selection (SFFS) method for channel selection.

    Parameters
    ----------
    kernel_func : function
        The classification kernel function which does feature extraction
        and classification.
        Different functions  are used for MI, P300, SSVEP, etc.
    X : numpy.ndarray
        Training data for the classifier as trials of EEG data.
        3D array containing data with `float` type.

        shape = (`n_trials`,`n_channels`,`n_samples`)
    y : numpy.ndarray
        Training labels for the classifier.
        1D array.

        shape = (`n_trials`)
    channel_labels: list of `str`
        The set of channel labels corresponding to `n_channels`.
        A list of strings with length = `n_channels`.
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
    record_performance : bool
        Flag on whether or not to record performance metrics at each step.

    Returns
    -------
    new_channel_subset : list of `str`
        The new best channel subset from the list of `channel_labels`.
    self.clf : classifier
        The trained classification model.
    preds : numpy.ndarray
        The predictions from the model.
        1D array with the same shape as `y`.

        shape = (`n_trials`)
    accuracy : float
        The accuracy of the trained classification model.
    precision : float
        The precision of the trained classification model.
    recall : float
        The recall of the trained classification model.
    results_df : pandas.DataFrame
        A dataframe containing the performance metrics at each step.


    """
    results_df = pd.DataFrame(
        columns=[
            "Step",
            "Time",
            "N Channels",
            "Channel Subset",
            "Unique Combinations Tested in Step",
            "Accuracy",
            "Precision",
            "Recall",
        ]
    )
    step = 1

    start_time = time.time()

    n_trials, n_channels, n_samples = X.shape
    sffs_subset = []
    all_sets_tried = []  # set of all channels that have been tried

    for i, c in enumerate(channel_labels):
        if c in initial_channels:
            sffs_subset.append(i)

    performance_at_nchannels = np.zeros(len(channel_labels))
    performance_at_nchannels[: min_channels - 1] = np.inf
    best_subset_at_nchannels = [0] * len(channel_labels)

    previous_performance = 0

    stop_criterion = False

    # Get the performance of the initial subset, if possible
    try:
        (
            initial_model,
            initial_preds,
            initial_accuracy,
            initial_precision,
            initial_recall,
        ) = kernel_func(X[:, sffs_subset, :], y)
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

        performance_at_nchannels[len(initial_channels) - 1] = initial_performance
        best_subset_at_nchannels[len(initial_channels) - 1] = initial_channels

    # If not possible then set the initial performance to 0
    except ValueError:
        best_channel_subset = []
        best_model = None
        best_performance = 0
        best_preds = []
        best_accuracy = 0
        best_precision = 0
        best_recall = 0

    preds = []
    accuracy = 0
    precision = 0
    recall = 0

    pass_stopping_criterion = False

    # TODO Test the initial subset

    while stop_criterion is False:
        sets_to_try = []
        X_to_try = []
        for c in range(n_channels):
            if c not in sffs_subset:
                set_to_try = sffs_subset.copy()
                set_to_try.append(c)
                sets_to_try.append(set_to_try)

                # Get the new subset of data
                new_subset_data = np.zeros((n_trials, len(set_to_try), n_samples))
                for subset_idx, channel_number in enumerate(set_to_try):
                    channel_data = X[:, channel_number, :]
                    new_subset_data[:, subset_idx, :] = channel_data

                # Add to list of all subsets of X to try
                X_to_try.append(new_subset_data)

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
            logger.warning("Performance metric invalid, defaulting to accuracy")
            performances = accuracies

        best_round_performance = np.max(performances)
        best_set_index = accuracies.index(best_round_performance)

        sffs_subset = sets_to_try[best_set_index]
        new_channel_subset = [channel_labels[c] for c in sffs_subset]
        model = models[best_set_index]
        preds = predictions[best_set_index]
        accuracy = accuracies[best_set_index]
        precision = precisions[best_set_index]
        recall = recalls[best_set_index]

        current_performance = best_round_performance

        logger.debug("Removed a channel")
        logger.debug("New subset: %s", new_channel_subset)
        logger.debug("Accuracy: %s", accuracy)
        logger.debug("Accuracies: %s", accuracies)

        # If this is the best perfomance at n_channels
        if performance_at_nchannels[len(sffs_subset) - 1] < current_performance:
            performance_at_nchannels[len(sffs_subset) - 1] = current_performance
            best_subset_at_nchannels[len(sffs_subset) - 1] = sffs_subset

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
        elif current_performance >= best_performance and len(new_channel_subset) < len(
            best_channel_subset
        ):
            best_channel_subset = new_channel_subset
            best_model = model
            best_performance = current_performance
            best_preds = preds
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall

        if record_performance:
            new_channel_subset.sort()
            results_df.loc[step] = [
                step,
                time.time() - start_time,
                len(new_channel_subset),
                "".join(new_channel_subset),
                len(sets_to_try),
                accuracy,
                precision,
                recall,
            ]

        step += 1

        # Conditional Exclusion
        while stop_criterion is False:
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

                # Get the new subset of data
                new_subset_data = np.zeros((n_trials, len(set_to_try), n_samples))
                for subset_idx, channel_number in enumerate(set_to_try):
                    channel_data = X[:, channel_number, :]
                    new_subset_data[:, subset_idx, :] = channel_data

                # Add to list of all subsets of X to try
                X_to_try.append(new_subset_data)

            if X_to_try == []:
                break

            # run the kernel on the new sets
            outputs = Parallel(n_jobs=n_jobs)(
                delayed(kernel_func)(Xtest, y) for Xtest in X_to_try
            )

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
                logger.warning("Performance metric invalid, defaulting to accuracy")
                performances = accuracies

            best_round_performance = np.max(performances)
            best_set_index = accuracies.index(best_round_performance)

            # if performance is better at the resultant channel length
            if (
                performance_at_nchannels[length_of_resultant_set - 1]
                < best_round_performance
            ):
                sffs_subset = sets_to_try[best_set_index]
                new_channel_subset = [channel_labels[c] for c in sffs_subset]
                model = models[best_set_index]
                preds = predictions[best_set_index]
                accuracy = accuracies[best_set_index]
                precision = precisions[best_set_index]
                recall = recalls[best_set_index]

                logger.debug("Added back a channel")
                logger.debug("New subset: %s", new_channel_subset)
                logger.debug("Accuracy: %s", accuracy)
                logger.debug("Accuracies: %s", accuracies)

                current_performance = best_round_performance

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
                elif current_performance >= best_performance and len(
                    new_channel_subset
                ) < len(best_channel_subset):
                    best_channel_subset = new_channel_subset
                    best_model = model
                    best_performance = current_performance
                    best_preds = preds
                    best_accuracy = accuracy
                    best_precision = precision
                    best_recall = recall

                if record_performance:
                    new_channel_subset.sort()
                    results_df.loc[step] = [
                        step,
                        time.time() - start_time,
                        len(new_channel_subset),
                        "".join(new_channel_subset),
                        len(sets_to_try),
                        accuracy,
                        precision,
                        recall,
                    ]
                    step += 1

                performance_at_nchannels[length_of_resultant_set - 1] = (
                    current_performance
                )
                best_subset_at_nchannels[length_of_resultant_set - 1] = sffs_subset

            # if no performance gains, then stop conditional exclusion
            else:
                break

            # Check stopping criterion
            if pass_stopping_criterion is False:
                stop_criterion = __check_stopping_criterion(
                    time.time() - start_time,
                    len(new_channel_subset),
                    p_delta,
                    max_time,
                    min_channels,
                    max_channels,
                    performance_delta,
                )

        if pass_stopping_criterion:
            pass_stopping_criterion = False
            continue
        else:
            stop_criterion = __check_stopping_criterion(
                time.time() - start_time,
                len(new_channel_subset),
                p_delta,
                max_time,
                min_channels,
                max_channels,
                performance_delta,
            )

    new_channel_subset = [channel_labels[c] for c in sffs_subset]

    logger.debug("Best channel subset: %s", best_channel_subset)
    logger.debug("%s : %s", metric, best_performance)
    logger.debug("Time to optimal subset: %s s", time.time() - start_time)

    return (
        best_channel_subset,
        best_model,
        best_preds,
        best_accuracy,
        best_precision,
        best_recall,
        results_df,
    )
