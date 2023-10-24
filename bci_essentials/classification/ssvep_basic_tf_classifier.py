"""**SSVEP Basic Training-Free Classifier**

Classifies SSVEP based on relative bandpower, taking only the maximum.

"""

# Stock libraries
import os
import sys
import numpy as np
from scipy import signal

from bci_essentials.classification import Generic_classifier

# from bci_essentials.visuals import *
# from bci_essentials.signal_processing import *
# from bci_essentials.channel_selection import *

# Custom libraries
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


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

    def fit(self, print_fit=True, print_performance=True):
        """Fit the model.

        Parameters
        ----------
        print_fit : bool, *optional*
            Description of parameter `print_fit`.
            - Default is `True`.
        print_performance : bool, *optional*
            Description of parameter `print_performance`.
            - Default is `True`.

        Returns
        -------
        `None`
            Models created used in `predict()`.

        """
        print(
            "Oh deary me you must have mistaken me for another classifier which requires training"
        )
        print("I DO NOT NEED TRAINING.")
        print("THIS IS MY FINAL FORM")

    def predict(self, X, print_predict):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Description of parameter `X`.
            If array, state size and type. E.g.
            3D array containing data with `float` type.

            shape = (`1st_dimension`,`2nd_dimension`,`3rd_dimension`)
        print_predict : bool
            Description of parameter `print_predict`.

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
            print("setting up the training free classifier")

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
