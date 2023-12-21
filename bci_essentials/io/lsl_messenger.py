from pylsl import StreamInfo, StreamOutlet, IRREGULAR_RATE

from .messenger import Messenger
from ..classification.generic_classifier import Prediction

__all__ = ["LslMessenger"]


class LslMessenger(Messenger):
    """A Messenger object for sending event messages to an LSL outlet."""

    def __init__(self):
        """Create an LslMessenger object.

        If the LSL outlet cannot be created, an exception is raised."""
        try:
            info = StreamInfo(
                name="PythonResponse",
                type="BCI",
                channel_count=1,
                nominal_srate=IRREGULAR_RATE,
                channel_format="string",
                source_id="pyp30042",
            )
            self.__outlet = StreamOutlet(info)
            self.__outlet.push_sample(["This is the python response stream"])
        except Exception:
            raise Exception("LslMessenger: could not create outlet")

    def ping(self):
        self.__outlet.push_sample(["ping"])

    def marker_received(self, marker):
        self.__outlet.push_sample(["marker received : {}".format(marker)])

    def prediction(self, prediction: Prediction):
        # probability isn't used by Unity at this time
        self.__outlet.push_sample(["{}".format(prediction.labels)])
