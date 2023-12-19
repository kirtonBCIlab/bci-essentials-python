"""
A module for processing resting state data.

The EEG data inputs for each function are either a single windows or
a set of windows.
- For single windows, inputs are of the shape `C x S`, where:
    - C = number of channels
    - S = number of samples
- For multiple windows, inputs are of the shape `W x C x S`, where:
    - W = number of windows
    - C = number of channels
    - S = number of samples

"""

import numpy as np
import scipy.signal

from matplotlib import pyplot as plt

from .utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


def get_shape(data):
    """Get the shape of the input data.

    Parameters
    ----------
    data : numpy.ndarray
        Window(s) of resting state EEG data.
        2D or 3D array containing data with `float` type.

        shape = (`C_channels`,`S_samples`) OR
        (`W_windows`,`C_channels`,`S_samples`)

    Returns
    -------
    W : int
        Number of windows.
    C : int
        Number of channels.
    S : int
        Number of samples.

    """
    try:
        W, C, S = np.shape(data)
    except Exception:
        C, S = np.shape(data)
        W = 1

    return W, C, S


# This function is never used at the moment, anywhere in the code. Renaming to more clear convention on it being a public function.
def get_bandpower(data, fs, fmin, fmax, normalization=None):
    """Get the bandpower of a window of EEG.

    Parameters
    ----------
    data : numpy.ndarray
        A single resting state EEG window
        2D array containing data with `float` type.

        shape = (`C_channels`,`S_samples`)
    fs : float
        EEG sampling frequency.
    fmin : float
        Lower frequency bound.
    fmax : float
        Upper frequency bound.
    normalization : string, *optional*
        Method for normalization.
        - `"norm"` for divide by norm.
        - `"sum"` for divide by sum.
        - Default is `None` and no normalization is done.

    Returns
    -------
    power : numpy.ndarray
        Bandpower of frequency band.

        shape = (`C_channels`)

    """
    nchannels, nsamples = data.shape

    f, Pxx = scipy.signal.welch(data, fs=fs)

    # Normalization
    if normalization == "norm":
        Pxx = np.divide(Pxx, np.tile(np.linalg.norm(Pxx, axis=1), (len(f), 1)).T)
    if normalization == "sum":
        Pxx = Pxx / Pxx.sum(axis=1).reshape((Pxx.shape[0], 1))

    ind_local_min = np.argmax(f > fmin) - 1
    ind_local_max = np.argmax(f > fmax) - 1

    power = np.zeros([nchannels])
    for i in range(nchannels):
        power[i] = np.trapz(
            Pxx[i, ind_local_min:ind_local_max], f[ind_local_min:ind_local_max]
        )

    return power


# Alpha Peak
def get_alpha_peak(data, alpha_min=8, alpha_max=12, plot_psd=False):
    """Get the alpha peak based on the all channel median PSD.

    Parameters
    ----------
    data : numpy.ndarray
        Resting state EEG window with eyes closed.
        3D array containing data with `float` type.

        shape = (`W_windows`,`C_channels`,`S_samples`)
    alpha_min : float, *optional*
        Lowest possible value of alpha peak (Hz)
        - Default is `8`.
    alpha_max : float, *optional*
        Highest possible value of alpha peak (Hz)
        - Default is `12`.
    plot_psd : bool, *optional*
        Plot the PSD for inspection.
        - Default is `False`.

    Returns
    -------
    alpha_peaks : numpy.ndarray
        The peak alpha frequency (in Hz) for each window.
    """

    fs = 256

    W, C, S = get_shape(data)

    # Create alpha_peaks of length W
    alpha_peaks = np.zeros(W)

    for win in range(W):
        # Get the current window
        current_window = data[win, :, :]

        # Calculate PSD using Welch's method, nfft = nsamples
        f, Pxx = scipy.signal.welch(current_window, fs=fs, nperseg=S)

        # Limit f, Pxx to the band of interest
        ind_min = scipy.argmax(f > alpha_min) - 1
        ind_max = scipy.argmax(f > alpha_max) - 1

        f = f[ind_min:ind_max]
        Pxx = Pxx[:, ind_min:ind_max]

        alpha_peaks[win] = f[np.argmax(np.median(Pxx, axis=0))]
        logger.info("Alpha peak of window %s is %s", win, alpha_peaks[win])

        if plot_psd:
            nrows = int(np.ceil(np.sqrt(C)))
            ncols = int(np.ceil(np.sqrt(C)))

            fig, axs = plt.subplots(nrows, ncols, figsize=(10, 8))
            # fig.suptitle("Some PSDs")
            for r in range(nrows):
                for c in range(ncols):
                    ch = (ncols * r) + c
                    axs[r, c].set_title(ch)
                    axs[r, c].plot(f, Pxx[ch, :])

            plt.show()

            plt.figure()
            plt.plot(f, np.median(Pxx, axis=0))

            plt.show()

    return alpha_peaks


# Bandpower features
def get_bandpower_features(data, fs, transition_freqs=[0, 4, 8, 12, 30]):
    """Get bandpower features.

    Parameters
    ----------
    data : numpy.ndarray
        Windows of resting state EEG data.
        3D array containing data with `float` type.

        shape = (`W_windows`,`C_channels`,`S_samples`)
    fs : float
        Sampling frequency (Hz).
    transition_freqs : array-like, *optional*
        The transition frequencies of the desired power bands.
        The first value is the lower cutoff, the last value is the
        upper cutoff, the middle values are band transition frequencies.
        - Default is `[0, 4, 8, 12, 30]`.

    Returns
    -------
    abs_bandpower : numpy.ndarray
        A numpy array of the absolute bandpower of provided bands.
        The last item is the total bandpower. The array length is equal
        to the length of `transition_freqs`.
    rel_bandpower : numpy.ndarray
        A numpy array of the relative bandpower of provided bands with
        respect to the entire region of interest from `transition_freqs[0]`
        to `transition_freqs[-1]`. The last item is the total relative
        bandpower and should always equal `1`. The array length is equal to
        the length of `transition_freqs`.
    rel_bandpower_mat : array_like
        A numpy array of the relative bandpower of provided bands such that
        location `(R,C)` corresponds to the power of `R` relative to `C`.
        The final row and column correspond to the cumulative power of
        all bands such that `(R,-1)` is `R` relative to all bands

    """
    # Get Shape
    W, C, S = get_shape(data)

    # Initialize
    abs_bandpower = np.zeros((len(transition_freqs), W))
    rel_bandpower = np.zeros((len(transition_freqs), W))
    rel_bandpower_mat = np.zeros((len(transition_freqs), len(transition_freqs), W))

    # for each window
    for win in range(W):
        # Get the current window
        current_window = data[win, :, :]

        # Calculate PSD using Welch's method
        f, Pxx = scipy.signal.welch(current_window, fs=fs)

        # Limit f, Pxx to the band of interest
        ind_global_min = scipy.argmax(f > min(transition_freqs)) - 1
        ind_global_max = scipy.argmax(f > max(transition_freqs)) - 1

        f = f[ind_global_min:ind_global_max]
        Pxx = Pxx[:, ind_global_min:ind_global_max]

        # Calculate the absolute power of each band
        for tf in range(len(transition_freqs)):
            # The last item is the total
            if tf == len(transition_freqs) - 1:
                abs_bandpower[tf, win] = np.sum(abs_bandpower[:tf, win])
                continue

            fmin = transition_freqs[tf]
            fmax = transition_freqs[tf + 1]

            ind_local_min = np.argmax(f > fmin) - 1
            ind_local_max = np.argmax(f > fmax) - 1

            # Get power for each channel
            abs_power = np.zeros([C])
            # norm_power = np.zeros([C])
            for ch in range(C):
                abs_power[ch] = np.trapz(
                    Pxx[ch, ind_local_min:ind_local_max], f[ind_local_min:ind_local_max]
                )

            # Median across all channels
            abs_bandpower[tf, win] = np.median(abs_power)

        rel_bandpower[:, win] = abs_bandpower[:, win] / abs_bandpower[-1, win]

        # Calculate the relative power of each band
        for tf1 in range(len(transition_freqs)):
            for tf2 in range(len(transition_freqs)):
                rel_bandpower_mat[tf1, tf2, win] = (
                    abs_bandpower[tf1, win] / abs_bandpower[tf2, win]
                )

    return abs_bandpower, rel_bandpower, rel_bandpower_mat
