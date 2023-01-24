"""
Signal Processing Tools

"""

# Signal processing tools for processing windows OR decision blocks

# For windows:
# Inputs are N x M x P where N = number of channels, M = number of samples, and P = number of windows, for single window P = 1
# Outputs are the same dimensions

# For decision blocks
# Inputs are N x M x P where N = number of channels, M = number of samples, and P = number of possible selections
# Outputs are the same dimensions

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt

# 

# def common_average_reference(data):
#     N,M,P = np.shape(data)
    
#     average = np.average(data,axis)

#     return new_data

def dc_reject(data):
    """DC Reject

    Filters out DC shifts in the data

    Parameters
    ----------
    data : numpy array 
        Windows of EEG data, nwindows X nchannels X nsamples

    Returns
    -------
    new_data : numpy array
        Windows of DC rejected EEG data, nwindows X nchannels X nsamples
    """

    try:
        N, M, P = np.shape(data)
    except:
        N, M =np.shape(data)
        P = 1

    new_data = np.ndarray(shape=(N, M, P), dtype=float)

    b = [1, -1]
    a = [1, -0.99]

    for p in range(0, P):
        for n in range(0, N):
            new_data[n,...,p] = signal.filtfilt(b,a,data[n,...,p])

    return new_data

def detrend(data):
    """Detrend

    Wrapper for the scipy.signal.detrend method

    Parameters
    ----------
    data : numpy array 
        Windows of EEG data, nwindows X nchannels X nsamples

    Returns
    -------
    new_data : numpy array
        Windows of detrended EEG data, nwindows X nchannels X nsamples
    """
    # detrends the windows using the numpy detrend function
    try:
        N, M, P = np.shape(data)
    except:
        N, M =np.shape(data)
        P = 1
        
    new_data = np.ndarray(shape=(N, M, P), dtype=float)

    for p in range(0,P):
        new_data[0:N,0:M,p] = signal.detrend(data[0:N,0:M,p], axis=1)

    return new_data

def lowpass(data, f_high, order, fsample):
    """Lowpass Filter

    Filters out frequencies above f_high with a Butterworth filter

    Parameters
    ----------
    data : numpy array 
        Windows of EEG data, nwindows X nchannels X nsamples
    f_high : float
        Upper corner frequency
    order : int
        Order of the filter
    fsample : float
        Sampling rate of signal

    Returns
    -------
    new_data : numpy array
        Windows of filtered EEG data, nwindows X nchannels X nsamples
    """
    try:
        N, M, P = np.shape(data)
    except:
        N, M =np.shape(data)
        P = 1
    
    Wn = f_high/(fsample/2)
    new_data = np.ndarray(shape=(N, M, P), dtype=float) 
    
    b, a = signal.butter(order, Wn, btype='lowpass')
    
    for p in range(0,P):
        new_data[0:N,0:M,p] = signal.filtfilt(b, a, data[0:N,0:M,p], axis=1, padlen=30)
        
    return new_data

def bandpass(data, f_low, f_high, order, fsample):
    """Bandpass Filter

    Filters out frequencies outside of f_low-f_high with a Butterworth filter

    Parameters
    ----------
    data : numpy array 
        Windows of EEG data, nwindows X nchannels X nsamples
    f_low : float
        Lower corner frequency    
    f_high : float
        Upper corner frequency
    order : int
        Order of the filter
    fsample : float
        Sampling rate of signal

    Returns
    -------
    new_data : numpy array
        Windows of filtered EEG data, nwindows X nchannels X nsamples
    """
    Wn = [f_low/(fsample/2), f_high/(fsample/2)]
    b, a = signal.butter(order, Wn, btype='bandpass')

    try:
        P, N, M = np.shape(data)

        # reshape to N,M,P
        data_reshape = np.swapaxes(np.swapaxes(data, 1, 2), 0, 2)

        new_data = np.ndarray(shape=(N, M, P), dtype=float) 
        for p in range(0,P):
            new_data[0:N,0:M,p] = signal.filtfilt(b, a, data_reshape[0:N,0:M,p], 
                                                    axis=1, 
                                                    padlen=30)

        new_data = np.swapaxes(np.swapaxes(new_data, 0, 2), 1, 2)
        return new_data
        
    except:
        N, M = np.shape(data)

        new_data = np.ndarray(shape=(N, M), dtype=float) 
        new_data = signal.filtfilt(b, a, data, axis=1, padlen=0)

        return new_data

def notchfilt(data, fsample, Q=30, fc=60):
    """Notch Filter

    Notch filter for removing specific frequency components.

    Parameters
    ----------
    data : numpy array 
        Windows of EEG data, nwindows X nchannels X nsamples
    fsample : float
        Sampling rate of signal
    Q : float
        Quality factor. Dimensionless parameter that characterizes notch filter 
        -3 dB bandwidth bw relative to its center frequency, Q = w0/bw.
    fc : float
        Frequency of notch

    Returns
    -------
    new_data : numpy array
        Windows of filtered EEG data, nwindows X nchannels X nsamples
    """


    b, a = signal.iirnotch(fc, Q, fsample)

    try:
        N, M, P = np.shape(data)
        new_data = np.ndarray(shape=(N, M, P), dtype=float) 
        for p in range(0,P):
            new_data[0:N,0:M,p] = signal.filtfilt(b, a, data[0:N,0:M,p], axis=1, padlen=30)
        return new_data
        
    except:
        N, M = np.shape(data)
        new_data = np.ndarray(shape=(N, M), dtype=float) 
        new_data = signal.filtfilt(b, a, data, axis=1, padlen=30)
        return new_data
