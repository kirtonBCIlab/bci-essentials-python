import numpy as np
import unittest

from bci_essentials.signal_processing import (
    lowpass,
    highpass,
    bandpass,
    notch,
    lico,
    smote,
)
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger(name="test_signal_processing")


class TestLoadData(unittest.TestCase):
    def test_bandpass(self):
        # Generate test data
        fsample = 1000
        f_low = 10
        f_high = 20
        order = 5

        # Create a time vector of length 2 and interval 1/fsample
        t = np.arange(0, 1, 1 / fsample)

        # Create a 2D and 3D array of zeros
        example_2D = np.zeros((2, len(t)))
        example_3D = np.zeros((3, 2, len(t)))

        # Create a signal with same length as t and even frequency distribution from 1 to 70Hz
        frequency_components = np.arange(1, 70, 1)
        np.random.seed(0)
        random_all = np.zeros(len(t))
        random_band = np.zeros(len(t))

        # Add random sine waves to random_all
        random_phases = np.random.randn(len(frequency_components))
        for i, f in enumerate(frequency_components):
            random_all += np.sin(2 * np.pi * f * t + random_phases[i])

            # Add random sine waves to random_low if they are below f_critical
            if f > f_low and f < f_high:
                random_band += np.sin(2 * np.pi * f * t + random_phases[i])

        # Add random_all to each channel in the 2D arrays
        for i in range(0, 2):
            example_2D[i, :] = random_all

        # Add random_all to each channel in the 3D arrays
        for i in range(0, 3):
            for j in range(0, 2):
                example_3D[i, j, :] = random_all

        # Filter the 2D and 3D arrays
        example_2D_band = bandpass(example_2D, f_low, f_high, order, fsample)
        example_3D_band = bandpass(example_3D, f_low, f_high, order, fsample)

        # Check that output has correct shape
        assert example_2D_band.shape == example_2D.shape
        assert example_3D_band.shape == example_3D.shape

        # Calculate the MSE between example_2D_low[0,:] and both random_low and random_all
        mse_2d_band = np.mean(np.square(example_2D_band[0, :] - random_band))
        mse_2d_all = np.mean(np.square(example_2D_band[0, :] - random_all))

        # Calculate the MSE between example_3D_low[0,0,:] and both random_low and random_all
        mse_3d_band = np.mean(np.square(example_3D_band[0, 0, :] - random_band))
        mse_3d_all = np.mean(np.square(example_3D_band[0, 0, :] - random_all))

        # # Log the MSEs
        logger.debug("MSE 2D band: %s", mse_2d_band)
        logger.debug("MSE 2D all: %s", mse_2d_all)
        logger.debug("MSE 3D band: %s", mse_3d_band)
        logger.debug("MSE 3D all: %s", mse_3d_all)

        # Check that output is correct
        assert mse_2d_band < mse_2d_all
        assert mse_3d_band < mse_3d_all

    def test_lowpass(self):
        # Generate test data
        fsample = 1000
        f_critical = 10
        order = 5

        # Create a time vector of length 2 and interval 1/fsample
        t = np.arange(0, 1, 1 / fsample)

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
        mse_2d_low = np.mean(np.square(example_2D_low[0, :] - random_low))
        mse_2d_all = np.mean(np.square(example_2D_low[0, :] - random_all))

        # Calculate the MSE between example_3D_low[0,0,:] and both random_low and random_all
        mse_3d_low = np.mean(np.square(example_3D_low[0, 0, :] - random_low))
        mse_3d_all = np.mean(np.square(example_3D_low[0, 0, :] - random_all))

        # # Log the MSEs
        logger.debug("MSE 2D low: %s", mse_2d_low)
        logger.debug("MSE 2D all: %s", mse_2d_all)
        logger.debug("MSE 3D low: %s", mse_3d_low)
        logger.debug("MSE 3D all: %s", mse_3d_all)

        # Check that output is correct
        assert mse_2d_low < mse_2d_all
        assert mse_3d_low < mse_3d_all

    def test_highpass(self):
        # Generate test data
        fsample = 1000
        f_critical = 60
        order = 5

        # Create a time vector of length 2 and interval 1/fsample
        t = np.arange(0, 1, 1 / fsample)

        # Create a 2D and 3D array of zeros
        example_2D = np.zeros((2, len(t)))
        example_3D = np.zeros((3, 2, len(t)))

        # Create a signal with same length as t and even frequency distribution from 0.01 to 70Hz
        frequency_components = np.arange(1, 70, 1)
        np.random.seed(0)
        random_all = np.zeros(len(t))
        random_high = np.zeros(len(t))

        # Add random sine waves to random_all
        random_phases = np.random.randn(len(frequency_components))
        for i, f in enumerate(frequency_components):
            random_all += np.sin(2 * np.pi * f * t + random_phases[i])

            # Add random sine waves to random_high if they are above f_critical
            if f > f_critical:
                random_high += np.sin(2 * np.pi * f * t + random_phases[i])

        # Add random_all to each channel in the 2D arrays
        for i in range(0, 2):
            example_2D[i, :] = random_all

        # Add random_all to each channel in the 3D arrays
        for i in range(0, 3):
            for j in range(0, 2):
                example_3D[i, j, :] = random_all

        # Filter the 2D and 3D arrays
        example_2D_high = highpass(example_2D, f_critical, order, fsample)
        example_3D_high = highpass(example_3D, f_critical, order, fsample)

        # Check that output has correct shape
        assert example_2D_high.shape == example_2D.shape
        assert example_3D_high.shape == example_3D.shape

        # Calculate the MSE between example_2D_high[0,:] and both random_high and random_all
        mse_2d_high = np.mean(np.square(example_2D_high[0, :] - random_high))
        mse_2d_all = np.mean(np.square(example_2D_high[0, :] - random_all))

        # Calculate the MSE between example_3D_high[0,0,:] and both random_high and random_all
        mse_3d_high = np.mean(np.square(example_3D_high[0, 0, :] - random_high))
        mse_3d_all = np.mean(np.square(example_3D_high[0, 0, :] - random_all))

        # # Log the MSEs
        logger.debug("MSE 2D high: %s", mse_2d_high)
        logger.debug("MSE 2D all: %s", mse_2d_all)
        logger.debug("MSE 3D high: %s", mse_3d_high)
        logger.debug("MSE 3D all: %s", mse_3d_all)

        # Check that output is correct
        assert mse_2d_high < mse_2d_all
        assert mse_3d_high < mse_3d_all

    def test_notch(self):
        # Generate test data
        fsample = 1000
        f_notch = 60
        Q = 30

        # Create a time vector of length 2 and interval 1/fsample
        t = np.arange(0, 1, 1 / fsample)

        # Create a 2D and 3D array of zeros
        example_2D = np.zeros((2, len(t)))
        example_3D = np.zeros((3, 2, len(t)))

        # Create a signal with same length as t and even frequency distribution from 0.01 to 70Hz
        frequency_components = np.arange(1, 70, 1)
        np.random.seed(0)
        random_all = np.zeros(len(t))
        random_notch = np.zeros(len(t))

        # Add random sine waves to random_all
        random_phases = np.random.randn(len(frequency_components))
        for i, f in enumerate(frequency_components):
            random_all += np.sin(2 * np.pi * f * t + random_phases[i])

            # Add random sine waves to random_notch if they are above f_notch
            if f != f_notch:
                random_notch += np.sin(2 * np.pi * f * t + random_phases[i])

        # Add random_all to each channel in the 2D arrays
        for i in range(0, 2):
            example_2D[i, :] = random_all

        # Add random_all to each channel in the 3D arrays
        for i in range(0, 3):
            for j in range(0, 2):
                example_3D[i, j, :] = random_all

        # Filter the 2D and 3D arrays
        example_2D_notch = notch(example_2D, f_notch, Q, fsample)
        example_3D_notch = notch(example_3D, f_notch, Q, fsample)

        # Check that output has correct shape
        assert example_2D_notch.shape == example_2D.shape
        assert example_3D_notch.shape == example_3D.shape

        # Calculate the MSE between example_2D_notch[0,:] and both random_notch and random_all
        mse_2d_notch = np.mean(np.square(example_2D_notch[0, :] - random_notch))
        mse_2d_all = np.mean(np.square(example_2D_notch[0, :] - random_all))

        # Calculate the MSE between example_3D_notch[0,0,:] and both random_notch and random_all
        mse_3d_notch = np.mean(np.square(example_3D_notch[0, 0, :] - random_notch))
        mse_3d_all = np.mean(np.square(example_3D_notch[0, 0, :] - random_all))

        # # Log the MSEs
        logger.debug("MSE 2D notch: %s", mse_2d_notch)
        logger.debug("MSE 2D all: %s", mse_2d_all)
        logger.debug("MSE 3D notch: %s", mse_3d_notch)
        logger.debug("MSE 3D all: %s", mse_3d_all)

        # Check that output is correct
        assert mse_2d_notch < mse_2d_all
        assert mse_3d_notch < mse_3d_all

    def test_lico(self):
        logger.debug("Testing LICO function")

        # Test data settings
        n_trials = 10
        n_channels = 16
        fsample = 256
        duration = 10
        n_samples = fsample * duration

        # Generate random matrix
        X = np.random.randn(n_trials, n_channels, n_samples)

        # Generate labels
        percentage_class_1 = 0.2
        n_ones = int(n_trials * percentage_class_1)
        n_zeros = n_trials - n_ones
        y = np.array([0] * n_zeros + [1] * n_ones, dtype=int)

        # Test LICO expansion factor
        expansion_factor = 3
        [oversample_X, oversample_y] = lico(X, y, expansion_factor)

        # Log the shapes of the output arrays
        logger.debug("Original X shape: %s", X.shape)
        logger.debug("Original class distribution: %s", np.bincount(y).tolist())
        logger.debug("Oversampled X shape: %s", oversample_X.shape)
        logger.debug(
            "Oversampled class distribution: %s", np.bincount(oversample_y).tolist()
        )

        # Checks
        # Check that number of channels, samples and mayority class remains the same
        assert X.shape[1] == oversample_X.shape[1]
        assert X.shape[2] == oversample_X.shape[2]
        assert np.sum(oversample_y == 0) == np.sum(y == 0)

        # Check that minority classs was expanded by the expansion factor
        assert np.sum(oversample_y == 1) == np.sum(y == 1) * expansion_factor
        assert (
            oversample_y.shape[0] == oversample_X.shape[0]
        )  # Number of labels matches number of trials

    def test_smote(self):
        logger.debug("Testing SMOTE function")

        # Test data settings
        n_trials = 100
        n_channels = 16
        fsample = 256
        duration = 1
        n_samples = fsample * duration

        # Generate random matrix
        X = np.random.randn(n_trials, n_channels, n_samples)

        # Generate imbalanced labels (20% class 1, 80% class 0)
        percentage_class_1 = 0.2
        n_ones = int(n_trials * percentage_class_1)
        n_zeros = n_trials - n_ones
        y = np.array([0] * n_zeros + [1] * n_ones, dtype=int)

        # Test SMOTE expansion factor
        expansion_factor = 3
        k_neighbors = 5
        [oversample_X, oversample_y] = smote(X, y, expansion_factor, k_neighbors)

        # Log the shapes of the output arrays
        logger.debug("Original X shape: %s", X.shape)
        logger.debug("Original class distribution: %s", np.bincount(y).tolist())
        logger.debug("Oversampled X shape: %s", oversample_X.shape)
        logger.debug(
            "Oversampled class distribution: %s", np.bincount(oversample_y).tolist()
        )

        # Checks
        # Check that number of channels and samples remains the same
        assert X.shape[1] == oversample_X.shape[1]
        assert X.shape[2] == oversample_X.shape[2]

        # Check that majority class (0) remains unchanged
        assert np.sum(oversample_y == 0) == np.sum(y == 0)

        # Check that minority class was expanded (approximately) by the expansion factor
        # SMOTE may not exactly match the expansion factor due to its implementation
        expected_minority = np.sum(y == 1) * expansion_factor
        actual_minority = np.sum(oversample_y == 1)
        tolerance = 0.1 * expected_minority  # 20% tolerance
        assert np.abs(actual_minority - expected_minority) <= tolerance

        # Check that the number of labels matches the number of trials
        assert oversample_y.shape[0] == oversample_X.shape[0]

        # Check that all values are either 0 or 1
        assert set(np.unique(oversample_y)) <= {0, 1}


if __name__ == "__main__":
    unittest.main()
