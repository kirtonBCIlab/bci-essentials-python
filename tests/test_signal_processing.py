
import os
import sys
import numpy as np
from scipy import signal

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from bci_essentials.signal_processing import lowpass

def test_lowpass():
    # Generate test data
    fsample = 1000
    f_critical = 10
    order = 5

    # Create a time vector of length 2 and interval 1/fsample
    t = np.arange(0, 1, 1/fsample)

    # Create a 2D and 3D array of zeros
    example_2D = np.zeros((2, len(t)))
    example_3D = np.zeros((3, 2, len(t)))

    # Create a signal with same length as t and even frequency distribution from 0.01 to 70Hz
    frequency_components = np.arange(1, 70, 1)
    np.random.seed(0)
    random_all = np.zeros(len(t))
    random_low = np.zeros(len(t))

    # Add random sine waves to random_all
    random_phases = np.random.randn(len(frequency_components))
    for i, f in enumerate(frequency_components):
        random_all += np.sin(2 * np.pi * f * t + random_phases[i])
        
        # Add random sine waves to random_low if they are below f_critical
        if f < f_critical:
            random_low += np.sin(2 * np.pi * f * t + random_phases[i])

    # Add random_all to each channel in the 2D arrays
    for i in range(0, 2):
        example_2D[i, :] = random_all

    # Add random_all to each channel in the 3D arrays
    for i in range(0, 3):
        for j in range(0, 2):
            example_3D[i, j, :] = random_all

    # Filter the 2D and 3D arrays
    example_2D_low = lowpass(example_2D, f_critical, order, fsample)
    example_3D_low = lowpass(example_3D, f_critical, order, fsample)

    # Check that output has correct shape
    assert example_2D_low.shape == example_2D.shape
    assert example_3D_low.shape == example_3D.shape

    # Calculate the MSE between example_2D_low[0,:] and both random_low and random_all
    mse_2d_low = np.mean(np.square(example_2D_low[0,:] - random_low))
    mse_2d_all = np.mean(np.square(example_2D_low[0,:] - random_all))

    # Calculate the MSE between example_3D_low[0,0,:] and both random_low and random_all
    mse_3d_low = np.mean(np.square(example_3D_low[0,0,:] - random_low))
    mse_3d_all = np.mean(np.square(example_3D_low[0,0,:] - random_all))

    # Check that output is correct
    assert mse_2d_low < mse_2d_all
    assert mse_3d_low < mse_3d_all

test_lowpass()
