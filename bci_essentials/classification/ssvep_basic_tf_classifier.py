"""**SSVEP Basic Training-Free Classifier**

Classifies SSVEP based on relative bandpower, taking only the maximum.

"""

# Stock libraries
import numpy as np
from scipy import signal

# Import bci_essentials modules and methods
from ..classification.generic_classifier import GenericClassifier, Prediction
from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


class SsvepBasicTrainFreeClassifier(GenericClassifier):
    """SSVEP Basic Training-Free Classifier class
    (*inherits from GenericClassifier*).

    """

    def set_ssvep_settings(self, sampling_freq, target_freqs):
        """Set the SSVEP settings.

        Parameters
        ----------
        sampling_freq : int
            Description of parameter `sampling_freq`.
        target_freqs : list of `int`
            Description of parameter `target_freqs`.

        Returns
        -------
        `None`
            Models created used in `fit()`.

        """
        self.sampling_freq = sampling_freq
        self.target_freqs = target_freqs
        self.setup = False

    def fit(self):
        """Fit the model.

        Returns
        -------
        `None`
            Models created used in `predict()`.

        """
        logger.warning(
            "Oh deary me you must have mistaken me for another classifier which requires training"
        )
        logger.warning("I DO NOT NEED TRAINING.")
        logger.warning("THIS IS MY FINAL FORM")

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            3D array where shape = (trials, channels, samples)

        Returns
        -------
        prediction : Prediction
            Results of predict call containing the predicted class labels.  Probabilities
            are not available (empty list).

        """
        # get the shape
        n_trials, n_channels, n_samples = X.shape
        # The first time it is called it must be set up
        if self.setup is False:
            logger.info("Setting up the training free classifier")

            self.setup = True

        # Build one augmented channel, here by just adding them all together
        augmented_X = np.mean(X, axis=1)

        # Get the PSD estimate using Welch's method
        f, Pxx = signal.welch(augmented_X, fs=self.sampling_freq, nperseg=n_samples)

        # Get a vote for each trial
        prediction = np.zeros(n_trials)
        for trial in range(n_trials):
            # Get the frequency with the greatest PSD
            Pxx_of_f_bins = np.zeros(len(self.target_freqs))
            for i, tf in enumerate(self.target_freqs):
                # Get the closest frequency bin
                closest_freq_bin = np.argmin(np.abs(f - tf))

                Pxx_of_f_bins[i] = Pxx[trial][int(closest_freq_bin)]

            prediction[trial] = np.argmax(Pxx_of_f_bins)

        return Prediction(labels=prediction)
