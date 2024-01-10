"""
Signal processing tools for processing windows of EEG data.

The EEG data inputs can be 2D or 3D arrays.
- For single windows, inputs are of the shape `num_channels x num_samples`, where:
    - num_channels = number of channels
    - num_samples = number of samples
- For multiple windows, inputs are of the shape `num_windows x num_channels x num_samples`, where:
    - num_windows = number of windows
    - num_channels = number of channels
    - num_samples = number of samples

- Outputs are the same dimensions as input (windows, channels, samples)

"""
import numpy as np
from scipy import signal
import random
from .utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


def bandpass(data, f_low, f_high, order, fsample):
    """Bandpass Filter.

    Filters out frequencies outside of the range f_low to f_high with a
    Butterworth filter of specific order.

    Wraps the scipy.signal.butter and scipy.signal.filtfilt methods.

    Parameters
    ----------
    data : numpy.ndarray
        Windows of EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (num_windows, num_channels, num_samples) or (num_channels, num_samples)
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
        Windows of filtered EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (num_windows, num_channels, num_samples) or (num_channels, num_samples)
    """
    Wn = [f_low / (fsample / 2), f_high / (fsample / 2)]
    b, a = signal.butter(order, Wn, btype="bandpass")

    try:
        num_windows, num_channels, num_samples = np.shape(data)

        new_data = np.ndarray(
            shape=(num_windows, num_channels, num_samples), dtype=float
        )
        for window in range(0, num_windows):
            current_window = data[window, :, :]
            new_data[window, :, :] = signal.filtfilt(b, a, current_window, padlen=0)

        return new_data

    except ValueError:
        num_channels, num_samples = np.shape(data)

        new_data = np.ndarray(shape=(num_channels, num_samples), dtype=float)
        new_data = signal.filtfilt(b, a, data, padlen=0)

        return new_data


def lowpass(data, f_critical, order, fsample):
    """Lowpass Filter.

    Filters out frequencies above f_critical with a Butterworth filter of specific order.

    Wraps the scipy.signal.butter and scipy.signal.filtfilt methods.

    Parameters
    ----------
    data : numpy.ndarray
        Windows of EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (num_windows, num_channels, num_samples) or (num_channels, num_samples)
    f_critical : float
        Critical (cutoff) frequency.
    order : int
        Order of the filter.
    fsample : float
        Sampling rate of signal.

    Returns
    -------
    new_data : numpy.ndarray
        Windows of filtered EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (num_windows, num_channels, num_samples) or (num_channels, num_samples)
    """
    Wn = f_critical / (fsample / 2)
    b, a = signal.butter(order, Wn, btype="lowpass")

    try:
        num_windows, num_channels, num_samples = np.shape(data)

        new_data = np.ndarray(
            shape=(num_windows, num_channels, num_samples), dtype=float
        )
        for window in range(0, num_windows):
            for channel in range(0, num_channels):
                current_window = data[window, channel, :]
                new_data[window, channel, :] = signal.filtfilt(
                    b, a, current_window, padlen=0
                )

        return new_data

    except ValueError:
        num_channels, num_samples = np.shape(data)

        new_data = np.ndarray(shape=(num_channels, num_samples), dtype=float)
        new_data = signal.filtfilt(b, a, data, padlen=0)

        return new_data


def highpass(data, f_critical, order, fsample):
    """Highpass Filter.

    Filters out frequencies below f_critical with a Butterworth filter of specific order.

    Wraps the scipy.signal.butter and scipy.signal.filtfilt methods.

    Parameters
    ----------
    data : numpy.ndarray
        Windows of EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (num_windows, num_channels, num_samples) or (num_channels, num_samples)
    f_critical : float
        Critical (cutoff) frequency.
    order : int
        Order of the filter.
    fsample : float
        Sampling rate of signal.

    Returns
    -------
    new_data : numpy.ndarray
        Windows of filtered EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (num_windows, num_channels, num_samples) or (num_channels, num_samples)
    """
    Wn = f_critical / (fsample / 2)
    b, a = signal.butter(order, Wn, btype="highpass")

    try:
        num_windows, num_channels, num_samples = np.shape(data)

        new_data = np.ndarray(
            shape=(num_windows, num_channels, num_samples), dtype=float
        )
        for window in range(0, num_windows):
            current_window = data[window, :, :]
            new_data[window, :, :] = signal.filtfilt(b, a, current_window, padlen=0)

        return new_data

    except ValueError:
        num_channels, num_samples = np.shape(data)

        new_data = np.ndarray(shape=(num_channels, num_samples), dtype=float)
        new_data = signal.filtfilt(b, a, data, padlen=0)

        return new_data


def notch(data, f_notch, Q, fsample):
    """Notch Filter.

    Notch filter for removing specific frequency components.

    Parameters
    ----------
    data : numpy.ndarray
        Windows of EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (num_windows, num_channels, num_samples) or (num_channels, num_samples)
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
        Windows of filtered EEG data.
        3D array containing data with `float` type.

        shape = (`N_windows`,`M_channels`,`P_samples`)

    """

    b, a = signal.iirnotch(f_notch, Q, fsample)

    try:
        num_windows, num_channels, num_samples = np.shape(data)
        new_data = np.ndarray(
            shape=(num_windows, num_channels, num_samples), dtype=float
        )
        for window in range(0, num_windows):
            current_window = data[window, :, :]
            new_data[window, :, :] = signal.filtfilt(b, a, current_window, padlen=0)
        return new_data

    except Exception:
        num_channels, num_samples = np.shape(data)
        new_data = np.ndarray(shape=(num_channels, num_samples), dtype=float)
        new_data = signal.filtfilt(b, a, data, padlen=0)
        return new_data


def lico(X, y, expansion_factor=3, sum_num=2, shuffle=False):
    """Oversampling (linear combination oversampling (LiCO))

    Samples random linear combinations of existing epochs of X.

    This is broken, but I am also unsure if it deserves to be fixed. At the very least it probably belongs in a different file. -Brian

    Parameters
    ----------
    X : numpy.ndarray
        Windows of EEG data.
        3D array containing data with `float` type.

        shape = (num_windows, num_channels, num_samples)
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

    num_windows, num_channels, num_samples = true_X.shape
    logger.info("Shape of ERPs only: %s", true_X.shape)
    new_window = num_windows * np.round(expansion_factor - 1)
    new_X = np.zeros([new_window, num_channels, num_samples])
    for window in range(num_windows):
        for j in range(sum_num):
            random_epoch = true_X[random.choice(range(num_windows)), :, :]
            new_X[window, :, :] += random_epoch / sum_num

    over_X = np.append(X, new_X, axis=0)
    over_y = np.append(y, np.ones([new_window]))

    return over_X, over_y
