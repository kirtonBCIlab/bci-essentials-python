"""
Resting State

A module for processing resting state data.

The inputs for each function are either a single window (dimensions are
nchannels X nsamples) or a set of windows (dimensions are 
nchannels X nsamples X nwindows)
"""

import numpy as np
import scipy


def get_shape(data):
    """
    Get the shape of the input data.
    """
    try:
        N, M, P = np.shape(data)
    except:
        N, M = np.shape(data)
        P = 1

    return N, M, P


def bandpower(data, fs, fmin, fmax, normalization=None):
    """
    Get the bandpower of a window of EEG

    Parameters
    ----------
    data : array_like
        resting state EEG windows (nchannels X nsamples)
    fs : float
        EEG sampling frequency
    fmin : float
        lower frequency bound
    fmax : float
        upper frequency bound
    normalization : string
        method for normalization, "norm" for divide by norm, or "sum"
        for divide by sum

    Returns
    -------
    power : array_like
        bandpower of frequency band (nchannels)

    """
    nchannels, nsamples = data.shape

    f, Pxx = scipy.signal.welch(data, fs=fs)

    # Normalization
    if normalization == "norm":
        Pxx = np.divide(Pxx, np.tile(np.linalg.norm(Pxx, axis=1), (len(f), 1)).T)
    if normalization == "sum":
        Pxx = Pxx / Pxx.sum(axis=1).reshape((Pxx.shape[0], 1))

    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f > fmax) - 1

    power = np.zeros([nchannels])
    for i in range(nchannels):
        power[i] = scipy.trapz(Pxx[i, ind_min:ind_max], f[ind_min:ind_max])

    return power


# Alpha Peak
def get_alpha_peak(data):
    """
    Get the alpha peak.

    Parameters
    ----------
    data : array_like
        resting state EEG windows

    Returns
    -------
    peak : float
        the peak alpha frequency
    """

    N, M, P = get_shape(data)

    return


# Bandpower features
def get_bandpower_features(data, fs, transition_freqs=[0, 4, 8, 12, 30]):
    """
    Get bandpower features.

    Parameters
    ----------
    data : array_like
        resting state EEG windows
    fs : float
        sampling frequency (Hz)
    transition_freqs : array_like
        the transition frequencies of the desired power bands,
        the first value is the lower cutoff, the last value is the
        upper cutoff, the middle values are band transition frequencies

    Returns
    -------
    abs_bandpower : array_like
        a np array of the absolute bandpower of provided bands, length
        is equal to length of transition_freqs - 1
    norm_bandpower : array_like
        a np array of the normalized bandpower of provided bands, length
        is equal to length of transition_freqs - 1
    rel_bandpower_mat : array_like
        a np array of the relative bandpower of provided bands such
        location (R,C) is corresponds to the power of R relative to C,
        the final row and column correspond to the cumulative power of
        all bands such that (R,-1) is R relative to all bands
    """
    # Get Shape
    N, M, P = get_shape(data)

    # Initialize
    abs_bandpower = np.zeros((len(transition_freqs), P))
    norm_bandpower = np.zeros((len(transition_freqs), P))
    rel_bandpower_mat = np.zeros((len(transition_freqs), len(transition_freqs), P))

    # for each window
    for win in range(P):
        # Calculate PSD using Welch's method, nfft = nsamples
        f, Pxx = scipy.signal.welch(data[:, :, win], fs=fs, nperseg=M)

        # Calculate the absolute power of each band
        for tf in range(len(transition_freqs)):
            # The last item is the total
            if tf == len(transition_freqs) - 1:
                abs_bandpower[tf, win] = np.sum(abs_bandpower[: tf - 1, win])
                norm_bandpower[tf, win] = np.sum(norm_bandpower[: tf - 1, win])
                continue

            fmin = transition_freqs[tf]
            fmax = transition_freqs[tf + 1]

            # Normalize by sum
            norm_Pxx = Pxx / Pxx.sum(axis=1).reshape((Pxx.shape[0], 1))

            ind_min = scipy.argmax(f > fmin) - 1
            ind_max = scipy.argmax(f > fmax) - 1

            # Get power for each channel
            abs_power = np.zeros([N])
            norm_power = np.zeros([N])
            for ch in range(N):
                abs_power[ch] = scipy.trapz(
                    Pxx[ch, ind_min:ind_max], f[ind_min:ind_max]
                )
                norm_power[ch] = scipy.trapz(
                    norm_Pxx[ch, ind_min:ind_max], f[ind_min:ind_max]
                )

            # Average across all channels
            abs_bandpower[tf, win] = np.mean(abs_power)
            norm_bandpower[tf, win] = np.mean(norm_power)

        # Calculate the relative power of each band
        for tf1 in range(len(transition_freqs)):
            for tf2 in range(len(transition_freqs)):
                rel_bandpower_mat[tf1, tf2] = (
                    norm_bandpower[tf1, win] / norm_bandpower[tf2, win]
                )

    return abs_bandpower, norm_bandpower, rel_bandpower_mat
