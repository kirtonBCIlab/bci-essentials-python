"""**SSVEP Basic Training-Free Classifier**

Classifies SSVEP based on relative bandpower, taking only the maximum.

"""

# Stock libraries
import numpy as np
from scipy import signal

# Import bci_essentials modules and methods
from ..classification.generic_classifier import Generic_classifier
from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


class SSVEP_basic_tf_classifier(Generic_classifier):
    """SSVEP Basic Training-Free Classifier class
    (*inherits from Generic_classifier*).

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
            Description of parameter `X`.
            If array, state size and type. E.g.
            3D array containing data with `float` type.

            shape = (`1st_dimension`,`2nd_dimension`,`3rd_dimension`)

        Returns
        -------
        prediction : numpy.ndarray
            The predicted class labels.

            shape = (`1st_dimension`,)

        """
        # get the shape
        nwindows, nchannels, nsamples = X.shape
        # The first time it is called it must be set up
        if self.setup is False:
            logger.info("Setting up the training free classifier")

            self.setup = True

        # Build one augmented channel, here by just adding them all together
        X = np.mean(X, axis=1)

        # Get the PSD estimate using Welch's method
        f, Pxx = signal.welch(X, fs=self.sampling_freq, nperseg=nsamples)

        # Get a vote for each window
        prediction = np.zeros(nwindows)
        for w in range(nwindows):
            # Get the frequency with the greatest PSD
            f_bins = np.zeros(len(self.target_freqs))
            Pxx_of_f_bins = np.zeros(len(self.target_freqs))
            for i, tf in enumerate(self.target_freqs):
                # Get the closest frequency bin
                f_bins[i] = np.argmin(np.abs(f - tf))

                Pxx_of_f_bins[i] = Pxx[w][int(f_bins[i])]

            prediction[w] = np.argmax(Pxx_of_f_bins)

        return prediction
