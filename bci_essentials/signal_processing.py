"""
Signal processing tools for processing windows OR decision blocks.

The EEG data inputs for each function are either windows or
decision blocks.
- For windows, inputs are `N x M x P`, where:
    - N = number of windows (for a single window `N = 1`)
    - M = number of channels
    - P = number of samples
- For decision blocks, inputs are `N x M x P`, where:
    - N = number of possible selections
    - M = number of channels
    - P = number of samples
- Outputs are the same dimensions (`N x M x P`)

"""
import numpy as np
from scipy import signal
import random

#

# def common_average_reference(data):
#     N,M,P = np.shape(data)

#     average = np.average(data,axis)

#     return new_data


def dc_reject(data: np.ndarray):
    """DC Reject.

    Filters out DC shifts in the data.

    Parameters
    ----------
    data : numpy.ndarray
        Windows of EEG data.
        3D array containing data with `float` type.

        shape = (`N_windows`,`M_channels`,`P_samples`)

    Returns
    -------
    new_data : numpy.ndarray
        Windows of DC-rejected EEG data.
        3D array containing data with `float` type.

        shape = (`N_windows`,`M_channels`,`P_samples`)

    """
    try:
        N, M, P = np.shape(data)
    except Exception:
        N, M = np.shape(data)
        P = 1

    new_data = np.ndarray(shape=(N, M, P), dtype=float)

    b = [1, -1]
    a = [1, -0.99]

    for p in range(0, P):
        for n in range(0, N):
            new_data[n, ..., p] = signal.filtfilt(b, a, data[n, ..., p])

    return new_data


def detrend(data: np.ndarray):
    """Detrend.

    Wrapper for the scipy.signal.detrend method.

    Parameters
    ----------
    data : numpy.ndarray
        Windows of EEG data.
        3D array containing data with `float` type.

        shape = (`N_windows`,`M_channels`,`P_samples`)

    Returns
    -------
    new_data : numpy.ndarray
        Windows of detrended EEG data.
        3D array containing data with `float` type.

        shape = (`N_windows`,`M_channels`,`P_samples`)

    """
    # detrends the windows using the numpy detrend function
    try:
        N, M, P = np.shape(data)
    except Exception:
        N, M = np.shape(data)
        P = 1

    new_data = np.ndarray(shape=(N, M, P), dtype=float)

    for p in range(0, P):
        new_data[0:N, 0:M, p] = signal.detrend(data[0:N, 0:M, p], axis=1)

    return new_data


def lowpass(data: np.ndarray, f_high: float, order: int, fsample: float):
    """Lowpass Filter.

    Filters out frequencies above f_high with a Butterworth filter.

    Parameters
    ----------
    data : numpy.ndarray
        Windows of EEG data.
        3D array containing data with `float` type.

        shape = (`N_windows`,`M_channels`,`P_samples`)
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
        3D array containing data with `float` type.

        shape = (`N_windows`,`M_channels`,`P_samples`)

    """
    try:
        N, M, P = np.shape(data)
    except Exception:
        N, M = np.shape(data)
        P = 1

    Wn = f_high / (fsample / 2)
    new_data = np.ndarray(shape=(N, M, P), dtype=float)

    b, a = signal.butter(order, Wn, btype="lowpass")

    for p in range(0, P):
        new_data[0:N, 0:M, p] = signal.filtfilt(
            b, a, data[0:N, 0:M, p], axis=1, padlen=30
        )

    return new_data


def bandpass(data: np.ndarray, f_low: float, f_high: float, order: int, fsample: float):
    """Bandpass Filter.

    Filters out frequencies outside of the range f_low to f_high with a
    Butterworth filter.

    Parameters
    ----------
    data : numpy.ndarray
        Windows of EEG data.
        3D array containing data with `float` type.

        shape = (`N_windows`,`M_channels`,`P_samples`)
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
        3D array containing data with `float` type.

        shape = (`N_windows`,`M_channels`,`P_samples`)
    """
    Wn = [f_low / (fsample / 2), f_high / (fsample / 2)]
    b, a = signal.butter(order, Wn, btype="bandpass")

    try:
        P, N, M = np.shape(data)

        # reshape to N,M,P
        data_reshape = np.swapaxes(np.swapaxes(data, 1, 2), 0, 2)

        new_data = np.ndarray(shape=(N, M, P), dtype=float)
        for p in range(0, P):
            new_data[0:N, 0:M, p] = signal.filtfilt(
                b, a, data_reshape[0:N, 0:M, p], axis=1, padlen=30
            )

        new_data = np.swapaxes(np.swapaxes(new_data, 0, 2), 1, 2)
        return new_data

    except Exception:
        N, M = np.shape(data)

        new_data = np.ndarray(shape=(N, M), dtype=float)
        new_data = signal.filtfilt(b, a, data, axis=1, padlen=0)

        return new_data


def notchfilt(data: np.ndarray, fsample: float, Q: float = 30, fc: float = 60):
    """Notch Filter.

    Notch filter for removing specific frequency components.

    Parameters
    ----------
    data : numpy.ndarray
        Windows of EEG data.
        3D array containing data with `float` type.

        shape = (`N_windows`,`M_channels`,`P_samples`)
    fsample : float
        Sampling rate of signal.
    Q : float, *optional*
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth bw relative to its
        center frequency, Q = w0/bw.
        - Default is `30`.
    fc : float, *optional*
        Frequency of notch.
        - Default is `60`.

    Returns
    -------
    new_data : numpy.ndarray
        Windows of filtered EEG data.
        3D array containing data with `float` type.

        shape = (`N_windows`,`M_channels`,`P_samples`)

    """

    b, a = signal.iirnotch(fc, Q, fsample)

    try:
        N, M, P = np.shape(data)
        new_data = np.ndarray(shape=(N, M, P), dtype=float)
        for p in range(0, P):
            new_data[0:N, 0:M, p] = signal.filtfilt(
                b, a, data[0:N, 0:M, p], axis=1, padlen=30
            )
        return new_data

    except Exception:
        N, M = np.shape(data)
        new_data = np.ndarray(shape=(N, M), dtype=float)
        new_data = signal.filtfilt(b, a, data, axis=1, padlen=30)
        return new_data


def lico(
    X: np.ndarray,
    y: np.ndarray,
    expansion_factor: int = 3,
    sum_num: int = 2,
    shuffle: bool = False,
):
    """Oversampling (linear combination oversampling (LiCO))

    Samples random linear combinations of existing epochs of X.

    Parameters
    ----------
    X : numpy.ndarray
        The file location of the spreadsheet.
    y : numpy.ndarray
        A flag used to print the columns to the console.
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

    n, m, p = true_X.shape
    print("Shape of ERPs only ", true_X.shape)
    new_n = n * np.round(expansion_factor - 1)
    new_X = np.zeros([new_n, m, p])
    for i in range(n):
        for j in range(sum_num):
            new_X[i, :, :] += true_X[random.choice(range(n)), :, :] / sum_num

    over_X = np.append(X, new_X, axis=0)
    over_y = np.append(y, np.ones([new_n]))

    return over_X, over_y
