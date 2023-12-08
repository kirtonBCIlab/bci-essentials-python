from pylsl import StreamInfo, StreamOutlet, IRREGULAR_RATE

from .messenger import Messenger

__all__ = ["LslMessenger"]


class LslMessenger(Messenger):
    """A Messenger object that sends EEG_data events to an LSL outlet."""

    def __init__(self):
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

    def ping(self):
        self.__outlet.push_sample(["ping"])

    def marker_received(self, marker):
        self.__outlet.push_sample(["marker received : {}".format(marker)])

    def prediction(self, prediction):
        self.__outlet.push_sample(["{}".format(prediction)])
