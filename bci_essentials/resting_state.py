"""
A module for processing resting state data.

The EEG data inputs for each function are either a single trials or
a set of trials.
- For single trials, inputs are of the shape `n_channels x n_samples`, where:
    - n_channels = number of channels
    - n_samples = number of samples
- For multiple trials, inputs are of the shape `n_trials x n_channels x n_samples`, where:
    - n_trials = number of trials
    - n_channels = number of channels
    - n_samples = number of samples

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
        Trial(s) of resting state EEG data.
        2D or 3D array containing data with `float` type.

        shape = (`n_channels`,`n_samples`) OR
        (`n_trials`,`n_channels`,`n_samples`)

    Returns
    -------
    n_trials : int
        Number of trials.
    n_channels : int
        Number of channels.
    n_samples : int
        Number of samples.

    """
    try:
        n_trials, n_channels, n_samples = np.shape(data)
    except Exception:
        n_channels, n_samples = np.shape(data)
        n_trials = 1

    return n_trials, n_channels, n_samples


# This function is never used at the moment, anywhere in the code. Renaming to more clear convention on it being a public function.
def get_bandpower(data, fs, fmin, fmax, normalization=None):
    """Get the bandpower of a trial of EEG.

    Parameters
    ----------
    data : numpy.ndarray
        A single resting state EEG trial
        2D array containing data with `float` type.

        shape = (`n_channels`,`n_samples`)
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

        shape = (`n_channels`)

    """
    n_channels, n_samples = data.shape

    f, Pxx = scipy.signal.welch(data, fs=fs)

    # Normalization
    if normalization == "norm":
        Pxx = np.divide(Pxx, np.tile(np.linalg.norm(Pxx, axis=1), (len(f), 1)).T)
    if normalization == "sum":
        Pxx = Pxx / Pxx.sum(axis=1).reshape((Pxx.shape[0], 1))

    ind_local_min = np.argmax(f > fmin) - 1
    ind_local_max = np.argmax(f > fmax) - 1

    power = np.zeros([n_channels])
    for channel in range(n_channels):
        power[channel] = np.trapz(
            Pxx[channel, ind_local_min:ind_local_max], f[ind_local_min:ind_local_max]
        )

    return power


# Alpha Peak
def get_alpha_peak(data, alpha_min=8, alpha_max=12, plot_psd=False):
    """Get the alpha peak based on the all channel median PSD.

    Parameters
    ----------
    data : numpy.ndarray
        Resting state EEG trial with eyes closed.
        3D array containing data with `float` type.

        shape = (`n_trials`,`n_channels`,`n_samples`)
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
        The peak alpha frequency (in Hz) for each trial.
    """

    fs = 256

    n_trials, n_channels, n_samples = get_shape(data)

    # Create alpha_peaks of length n_trials
    alpha_peaks = np.zeros(n_trials)

    for trial in range(n_trials):
        # Get the current trial
        current_trial = data[trial, :, :]

        # Calculate PSD using Welch's method, nfft = n_samples
        f, Pxx = scipy.signal.welch(current_trial, fs=fs, nperseg=n_samples)

        # Limit f, Pxx to the band of interest
        ind_min = scipy.argmax(f > alpha_min) - 1
        ind_max = scipy.argmax(f > alpha_max) - 1

        f = f[ind_min:ind_max]
        Pxx = Pxx[:, ind_min:ind_max]

        alpha_peaks[trial] = f[np.argmax(np.median(Pxx, axis=0))]
        logger.info("Alpha peak of trial %s is %s", trial, alpha_peaks[trial])

        if plot_psd:
            nrows = int(np.ceil(np.sqrt(n_channels)))
            ncols = int(np.ceil(np.sqrt(n_channels)))

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
        Trials of resting state EEG data.
        3D array containing data with `float` type.

        shape = (`n_trials`,`n_channels`,`n_samples`)
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
    n_trials, n_channels, n_samples = get_shape(data)

    # Initialize
    abs_bandpower = np.zeros((len(transition_freqs), n_trials))
    rel_bandpower = np.zeros((len(transition_freqs), n_trials))
    rel_bandpower_mat = np.zeros(
        (len(transition_freqs), len(transition_freqs), n_trials)
    )

    # for each trial
    for trial in range(n_trials):
        # Get the current trial
        current_trial = data[trial, :, :]

        # Calculate PSD using Welch's method
        f, Pxx = scipy.signal.welch(current_trial, fs=fs)

        # Limit f, Pxx to the band of interest
        ind_global_min = scipy.argmax(f > min(transition_freqs)) - 1
        ind_global_max = scipy.argmax(f > max(transition_freqs)) - 1

        f = f[ind_global_min:ind_global_max]
        Pxx = Pxx[:, ind_global_min:ind_global_max]

        # Calculate the absolute power of each band
        for tf in range(len(transition_freqs)):
            # The last item is the total
            if tf == len(transition_freqs) - 1:
                abs_bandpower[tf, trial] = np.sum(abs_bandpower[:tf, trial])
                continue

            fmin = transition_freqs[tf]
            fmax = transition_freqs[tf + 1]

            ind_local_min = np.argmax(f > fmin) - 1
            ind_local_max = np.argmax(f > fmax) - 1

            # Get power for each channel
            abs_power = np.zeros([n_channels])
            # norm_power = np.zeros([n_channels])
            for ch in range(n_channels):
                abs_power[ch] = np.trapz(
                    Pxx[ch, ind_local_min:ind_local_max], f[ind_local_min:ind_local_max]
                )

            # Median across all channels
            abs_bandpower[tf, trial] = np.median(abs_power)

        rel_bandpower[:, trial] = abs_bandpower[:, trial] / abs_bandpower[-1, trial]

        # Calculate the relative power of each band
        for tf1 in range(len(transition_freqs)):
            for tf2 in range(len(transition_freqs)):
                rel_bandpower_mat[tf1, tf2, trial] = (
                    abs_bandpower[tf1, trial] / abs_bandpower[tf2, trial]
                )

    return abs_bandpower, rel_bandpower, rel_bandpower_mat
