"""


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
    Wn = [f_low/(fsample/2), f_high/(fsample/2)]
    b, a = signal.butter(order, Wn, btype='bandpass')

    try:
        P, N, M = np.shape(data)

        # reshape to N,M,P
        data_reshape = np.swapaxes(np.swapaxes(data, 1, 2), 0, 2)

        new_data = np.ndarray(shape=(N, M, P), dtype=float) 
        for p in range(0,P):
            new_data[0:N,0:M,p] = signal.filtfilt(b, a, data_reshape[0:N,0:M,p], axis=1, padlen=30)

        # Visualize the effect of the filter
        # fig, axs = plt.subplots(N)
        # for n in range(N):
        #     axs[n].plot(data_reshape[n,0:M,2], label="before")
        #     axs[n].plot(new_data[n,0:M,2], label="after")

        #     axs[n].legend()

        # # plt.plot(data_reshape[2,0:M,p], label="before")
        # # plt.plot(new_data[2,0:M,p], label="after")
        # # plt.legend()

        # # plt.show()
        # # plt.close()
        # fig.show()

        new_data = np.swapaxes(np.swapaxes(new_data, 0, 2), 1, 2)
        return new_data
        
    except:
        N, M = np.shape(data)
        new_data = np.ndarray(shape=(N, M), dtype=float) 
        new_data = signal.filtfilt(b, a, data, axis=1, padlen=30)

        # # Visualize the effect of the filter
        # fig, axs = plt.subplots(N)
        # for n in range(N):
        #     axs[n].plot(data[n,0:M], label="before")
        #     axs[n].plot(new_data[n,0:M], label="after")

        #     axs[n].legend()

        # # plt.show()
        # # plt.close()
        # fig.show()

        return new_data

def notchfilt(data, fsample, Q=30, fc=60):
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



def moving_average(data, window_length):
    print("moooooo")

# PIPELINE
# select processing steps
# def sp_pipeline(data, moving_average=0, bandpass=0):
#     num_operations = moving_average + bandpass

#     for i in range (1, num_operations):
#         if moving_average == i:
#             print("Doing a moving average filter")
#         if bandpass == i:
#             print("Doing a bandpass filter")
    

# def welch_psd(data, fsample):
#     try:
#         P, N, M = np.shape(data)

#         # reshape to N,M,P
#         data_reshape = np.swapaxes(np.swapaxes(data, 1, 2), 0, 2)

#         new_data = np.ndarray(shape=(N, M, P), dtype=float) 
#         for p in range(0,P):
#             new_data[0:N,0:M,p] = signal.filtfilt(b, a, data_reshape[0:N,0:M,p], axis=1, padlen=30)

#         # Visualize the effect of the filter
#         # fig, axs = plt.subplots(N)
#         # for n in range(N):
#         #     axs[n].plot(data_reshape[n,0:M,2], label="before")
#         #     axs[n].plot(new_data[n,0:M,2], label="after")

#         #     axs[n].legend()

#         # # plt.plot(data_reshape[2,0:M,p], label="before")
#         # # plt.plot(new_data[2,0:M,p], label="after")
#         # # plt.legend()

#         # # plt.show()
#         # # plt.close()
#         # fig.show()

#         new_data = np.swapaxes(np.swapaxes(new_data, 0, 2), 1, 2)
#         return new_data
        
#     except:
#         N, M = np.shape(data)
#         new_data = np.ndarray(shape=(N, M), dtype=float) 
#         new_data = signal.welch(data, fs=fsample, axis=1)

#         # # Visualize the effect of the filter
#         # fig, axs = plt.subplots(N)
#         # for n in range(N):
#         #     axs[n].plot(data[n,0:M], label="before")
#         #     axs[n].plot(new_data[n,0:M], label="after")

#         #     axs[n].legend()

#         # # plt.show()
#         # # plt.close()
#         # fig.show()

#         return new_data
