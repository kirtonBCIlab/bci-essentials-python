"""
Signal processing tools for processing trials of EEG data.

The EEG data inputs can be 2D or 3D arrays.
- For single trials, inputs are of the shape `n_channels x n_samples`, where:
    - n_channels = number of channels
    - n_samples = number of samples
- For multiple trials, inputs are of the shape `n_trials x n_channels x n_samples`, where:
    - n_trials = number of trials
    - n_channels = number of channels
    - n_samples = number of samples

- Outputs are the same dimensions as input (trials, channels, samples)

"""

import random
import functools
import numpy as np
from scipy import signal
from typing import Callable, Any
from .utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


def validate_filter_input(func: Callable) -> Callable:
    """Decorator to validate input data for filter functions."""

    @functools.wraps(func)
    def wrapper(data: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        try:
            if not isinstance(data, np.ndarray):
                raise ValueError(
                    f"Input data for {func.__name__} must be a numpy array"
                )

            if not (data.ndim in [2, 3]):
                raise ValueError(
                    f"Data shape for {func.__name__} must be 2D or 3D array"
                )

            return func(data, *args, **kwargs)
        except Exception as e:
            logger.ERROR(f"Error in {func.__name__}: {str(e)}")
            return data

    return wrapper


@validate_filter_input
def bandpass(data, f_low, f_high, order, fsample):
    """Bandpass Filter.

    Filters out frequencies outside of the range f_low to f_high with a
    Butterworth filter of specific order.

    Wraps the scipy.signal.butter and scipy.signal.sosfiltfilt methods.

    Parameters
    ----------
    data : numpy.ndarray
        Trials of EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    f_low : float
        Lower corner frequency.
    f_high : float
        Upper corner frequency.
    order : int
        Order of the filter.
    fsample : float
        Sampling rate of signal.

    Returns
    -------
    new_data : numpy.ndarray
        Trials of filtered EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples) or (n_channels, n_samples)

    """
    Wn = [f_low / (fsample / 2), f_high / (fsample / 2)]
    sos = signal.butter(order, Wn, btype="bandpass", output="sos")

    filtered_data = signal.sosfiltfilt(sos, data, padlen=0)

    return filtered_data


@validate_filter_input
def lowpass(data, f_critical, order, fsample):
    """Lowpass Filter.

    Filters out frequencies above f_critical with a Butterworth filter of specific order.

    Wraps the scipy.signal.butter and scipy.signal.sosfiltfilt methods.

    Parameters
    ----------
    data : numpy.ndarray
        Trials of EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    f_critical : float
        Critical (cutoff) frequency.
    order : int
        Order of the filter.
    fsample : float
        Sampling rate of signal.

    Returns
    -------
    new_data : numpy.ndarray
        Trials of filtered EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples) or (n_channels, n_samples)

    """
    Wn = f_critical / (fsample / 2)
    sos = signal.butter(order, Wn, btype="lowpass", output="sos")

    filtered_data = signal.sosfiltfilt(sos, data, padlen=0)

    return filtered_data


@validate_filter_input
def highpass(data, f_critical, order, fsample):
    """Highpass Filter.

    Filters out frequencies below f_critical with a Butterworth filter of specific order.

    Wraps the scipy.signal.butter and scipy.signal.sosfiltfilt methods.

    Parameters
    ----------
    data : numpy.ndarray
        Trials of EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    f_critical : float
        Critical (cutoff) frequency.
    order : int
        Order of the filter.
    fsample : float
        Sampling rate of signal.

    Returns
    -------
    new_data : numpy.ndarray
        Trials of filtered EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    """
    Wn = f_critical / (fsample / 2)
    sos = signal.butter(order, Wn, btype="highpass", output="sos")

    filtered_data = signal.sosfiltfilt(sos, data, padlen=0)

    return filtered_data


@validate_filter_input
def notch(data, f_notch, Q, fsample):
    """Notch Filter.

    Notch filter for removing specific frequency components.

    Parameters
    ----------
    data : numpy.ndarray
        Trials of EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    f_notch : float
        Frequency of notch.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth bw relative to its
        center frequency, Q = w0/bw.
    fsample : float
        Sampling rate of signal.

    Returns
    -------
    new_data : numpy.ndarray
        Trials of filtered EEG data.
        3D array containing data with `float` type.

        shape = (`n_trials`,`n_channels`,`n_samples`)

    """

    b, a = signal.iirnotch(f_notch, Q, fsample)
    filtered_data = signal.filtfilt(b, a, data, padlen=0)

    return filtered_data


def lico(X, y, expansion_factor=3, sum_num=2, shuffle=False):
    """Oversampling (linear combination oversampling (LiCO))

    Samples random linear combinations of existing epochs of X.

    This is broken, but I am also unsure if it deserves to be fixed. At the very least it probably belongs in a different file. -Brian

    Parameters
    ----------
    X : numpy.ndarray
        Trials of EEG data.
        3D array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples)
    y : numpy.ndarray
        Labels corresponding to X.
    expansion_factor : int, *optional*
        Number of times larger to make the output set over_X
        - Default is `3`.
    sum_num : int, *optional*
        Number of signals to be summed together
        - Default is `2`.

    Returns
    -------
    over_X : numpy.ndarray
        Oversampled X.
    over_y : numpy.ndarray
        Oversampled y.

    """
    true_X = X[y == 1]

    n_trials, n_channels, n_samples = true_X.shape
    logger.info("Shape of ERPs only: %s", true_X.shape)
    new_trial = n_trials * np.round(expansion_factor - 1)
    new_X = np.zeros([new_trial, n_channels, n_samples])
    for trial in range(n_trials):
        for j in range(sum_num):
            random_epoch = true_X[random.choice(range(n_trials)), :, :]
            new_X[trial, :, :] += random_epoch / sum_num

    over_X = np.append(X, new_X, axis=0)
    over_y = np.append(y, np.ones([new_trial]))

    return over_X, over_y


def random_oversampling(X, y, ratio):
    """Random Oversampling

    Randomly samples epochs of X to oversample the MINORITY class.
    Automatically determines which class is the MINORITY class.

    Parameters
    ----------
    X : numpy.ndarray [n_trials, n_channels, n_samples]
        Trials of EEG data.
        3D array containing data with `float` type.
    y : numpy.ndarray [n_trials]
        Labels corresponding to X.
    ratio : float
        Desired ratio of MINORITY class samples to majority class samples
        - ratio=1 means the number of MINORITY class samples will be equal to the number of majority class samples
        - ratio=0.5 means the number of MINORITY class samples will be half the number of majority class samples
        - ratio=2 means the number of MINORITY class samples will be twice the number of majority class samples

    Returns
    -------
    over_X : numpy.ndarray
        Oversampled X.
    over_y : numpy.ndarray
        Oversampled y.
    """
    # Find unique classes and their counts
    classes, counts = np.unique(y, return_counts=True)

    # Determine minority and majority classes
    minority_class = classes[np.argmin(counts)]
    n_minority = np.min(counts)
    n_majority = np.max(counts)

    # Get minority class samples
    minority_X = X[y == minority_class]

    # Calculate number of samples needed
    n_samples = int(n_majority * ratio) - n_minority

    # Generate new samples
    new_X = np.zeros([n_samples, X.shape[1], X.shape[2]])
    for i in range(n_samples):
        new_X[i, :, :] = minority_X[random.choice(range(n_minority)), :, :]

    over_X = np.append(X, new_X, axis=0)
    over_y = np.append(y, np.ones([n_samples]) * minority_class)

    return over_X, over_y


def random_undersampling(X, y, ratio):
    """Random Undersampling

    Randomly removes epochs of X to undersample the MAJORITY class.
    Automatically determines which class is the MAJORITYajority class.

    Parameters
    ----------
    X : numpy.ndarray [n_trials, n_channels, n_samples]
        Trials of EEG data.
        3D array containing data with `float` type.
    y : numpy.ndarray [n_trials]
        Labels corresponding to X.
    ratio : float
        Desired ratio of MAJORITY class samples to minority class samples.
        - ratio=1 means the number of MAJORITY class samples will be equal to the number of minority class samples
        - ratio=0.5 means the number of MAJORITY class samples will be half the number of minority class samples
        - ratio=2 means the number of MAJORITY class samples will be twice the number of minority class samples

    Returns
    -------
    under_X : numpy.ndarray
        Undersampled X.
    under_y : numpy.ndarray
        Undersampled y.
    """
    # Find unique classes and their counts
    classes, counts = np.unique(y, return_counts=True)

    # Determine minority and majority classes
    majority_class = classes[np.argmax(counts)]
    minority_class = classes[np.argmin(counts)]
    n_minority = np.min(counts)

    # Calculate number of majority samples to keep
    n_samples = int(n_minority * ratio)

    # Get indices of majority class samples
    majority_indices = np.where(y == majority_class)[0]

    # Randomly select indices to keep
    keep_indices = np.random.choice(majority_indices, size=n_samples, replace=False)

    # Get indices of minority class samples
    minority_indices = np.where(y == minority_class)[0]

    # Combine indices
    all_indices = np.concatenate([keep_indices, minority_indices])

    # Create undersampled datasets
    under_X = X[all_indices]
    under_y = y[all_indices]

    return under_X, under_y
