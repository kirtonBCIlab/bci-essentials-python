from abc import ABC, abstractmethod


class Messenger(ABC):
    """A Messenger object is used by EEG_data to send time stamped event messages.  For example,
    to acknowledge that a marker has been received, or to provide a prediction.

    TODO - should all of these messages take a timestamp?

    TODO - do we need trial start / end messages?
    """

    @abstractmethod
    def ping(self):
        pass

    @abstractmethod
    def marker_received(self, marker):
        pass

    @abstractmethod
    def prediction(self, prediction):
        pass
