"""
Resting State

A module for processing resting state data.

The inputs for each function are either a single window (dimensions are
nchannels X nsamples) or a set of windows (dimensions are 
nchannels X nsamples X nwindows)
"""

import numpy as np
import scipy.signal

from matplotlib import pyplot as plt


def get_shape(data):
    """
    Get the shape of the input data.
    """
    try:
        W, C, S = np.shape(data)
    except:
        C, S = np.shape(data)
        W = 1

    return W, C, S


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

    ind_local_min = np.argmax(f > fmin) - 1
    ind_local_max = np.argmax(f > fmax) - 1

    power = np.zeros([nchannels])
    for i in range(nchannels):
        power[i] = np.trapz(Pxx[i, ind_local_min:ind_local_max], f[ind_local_min:ind_local_max])

    return power


# Alpha Peak
def get_alpha_peak(data, alpha_min=8, alpha_max=12, plot_psd=False):
    """
    Get the alpha peak based on the all channel median PSD.

    Parameters
    ----------
    data : array_like
        resting state EEG window with eyes closed

    alpha_min : float (default = 8)
        lowest possible value of alpha peak (Hz)

    alpha_max : float (default = 12)
        highest possible value of alpha peak (Hz)

    plot_psd : bool (default = False)
        plot the PSD for inspection


    Returns
    -------
    peak : float
        the peak alpha frequency
    """

    fs=256

    W, C, S = get_shape(data)

    
    
    for win in range(W):
        # Calculate PSD using Welch's method, nfft = nsamples
        f, Pxx = scipy.signal.welch(data[win, :, :], fs=fs, nperseg=S)

        # Limit f, Pxx to the band of interest
        ind_min = scipy.argmax(f > alpha_min) - 1
        ind_max = scipy.argmax(f > alpha_max) - 1

        f = f[ind_min:ind_max]
        Pxx = Pxx[:, ind_min:ind_max]

        try:
            median_Pxx[win,:] = np.median(Pxx, axis=0)

        except:
            median_Pxx = np.zeros([W, len(f)])
            median_Pxx[win,:] = np.median(Pxx, axis=0)

        alpha_peak = f[np.argmax(np.median(Pxx, axis=0))]
        print("Alpha peak of window {} ".format(win), alpha_peak)

        if plot_psd:
            nrows = int(np.ceil(np.sqrt(C)))
            ncols = int(np.ceil(np.sqrt(C)))

            fig, axs = plt.subplots(nrows, ncols, figsize=(10,8))
            # fig.suptitle("Some PSDs")
            for r in range(nrows):
                for c in range(ncols):
                    ch = (ncols*r) + c
                    axs[r, c].set_title(ch)
                    axs[r, c].plot(f, Pxx[ch,:])

                    # # axs[r, c].set_ylim([-20, 20])
                    # if r == 0 and c == 0:
                    #     axs[r, c].legend(["Open", "Closed"])

            plt.show()

            plt.figure()
            plt.plot(f, np.median(Pxx, axis=0))

            plt.show()

    overall_alpha_peak = f[np.argmax(np.median(median_Pxx, axis=0))]
    print("Overall alpha peak:", overall_alpha_peak)



    return overall_alpha_peak


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
        a np array of the absolute bandpower of provided bands, last 
        item is the total bandpower, length is equal to length of 
        transition_freqs
    rel_bandpower : array_like
        a np array of the relative bandpower of provided bands with
        respect to the entire region of interest from transition_freqs[0]
        to transition_freqs[-1], last item is the total relative 
        bandpower and should always equal 1, length is equal to length 
        of transition_freqs
    rel_bandpower_mat : array_like
        a np array of the relative bandpower of provided bands such
        location (R,C) is corresponds to the power of R relative to C,
        the final row and column correspond to the cumulative power of
        all bands such that (R,-1) is R relative to all bands
    """
    # Get Shape
    W, C, S = get_shape(data)

    # Initialize
    abs_bandpower = np.zeros((len(transition_freqs), W))
    rel_bandpower = np.zeros((len(transition_freqs), W))
    rel_bandpower_mat = np.zeros((len(transition_freqs), len(transition_freqs), W))

    # for each window
    for win in range(W):
        # Calculate PSD using Welch's method
        f, Pxx = scipy.signal.welch(data[win, :, :], fs=fs)

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
            norm_power = np.zeros([C])
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
