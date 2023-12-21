from abc import ABC, abstractmethod

from ..classification.generic_classifier import Prediction


class Messenger(ABC):
    """A Messenger object is used by EEG_data to send event messages.  For example,
    to acknowledge that a marker has been received, or to provide a prediction.
    """

    @abstractmethod
    def ping(self):
        """Indicate that sender is alive"""
        pass

    @abstractmethod
    def marker_received(self, marker):
        """Acknowledge that a marker was processed"""
        pass

    @abstractmethod
    def prediction(self, prediction: Prediction):
        """Send latest prediction"""
        pass
