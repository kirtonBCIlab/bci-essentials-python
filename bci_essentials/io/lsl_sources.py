from pylsl import StreamInlet, resolve_byprop, FOREVER

from .sources import MarkerSource, EegSource
from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)

__all__ = ["LslMarkerSource", "LslEegSource"]


class LslMarkerSource(MarkerSource):
    def __init__(self, timeout: float = FOREVER):
        """Create a MarkerSource object that obtains markers from an LSL outlet

        Parameters
        ----------
        timeout : float, *optional*
            How long to wait for marker outlet stream to be discovered.  If no stream
            is discovered, an Exception is raised.  By default init will wait forever.
        """
        try:
            marker_stream = resolve_byprop(
                "type", "LSL_Marker_Strings", timeout=timeout
            )
            self.__inlet = StreamInlet(marker_stream[0], processing_flags=0)
            self.__info = self.__inlet.info()
        except Exception:
            raise Exception("LslMarkerSource: could not create inlet")

    @property
    def name(self) -> str:
        return self.__info.name()

    def get_markers(self) -> tuple[list[list], list]:
        return pull_from_lsl_inlet(self.__inlet)

    def time_correction(self) -> float:
        return self.__inlet.time_correction()


class LslEegSource(EegSource):
    def __init__(self, timeout: float = FOREVER):
        """Create a MarkerSource object that obtains EEG from an LSL outlet

        Parameters
        ----------
        timeout : float, *optional*
            How long to wait for marker outlet stream to be discovered.  If no stream
            is discovered, an Exception is raised.  By defalut init will wait forever.
        """
        try:
            eeg_stream = resolve_byprop("type", "EEG", timeout=timeout)
            self.__inlet = StreamInlet(eeg_stream[0], processing_flags=0)
            self.__info = self.__inlet.info()
        except Exception:
            raise Exception("LslEegSource: could not create inlet")

    @property
    def name(self) -> str:
        return self.__info.name()

    @property
    def fsample(self) -> float:
        return self.__info.nominal_srate()

    @property
    def nchannels(self) -> int:
        return self.__info.channel_count()

    @property
    def channel_types(self) -> list[str]:
        return self.__get_channel_properties("type")

    @property
    def channel_units(self) -> list[str]:
        return self.__get_channel_properties("unit")

    @property
    def channel_labels(self) -> list[str]:
        return self.__get_channel_properties("name")

    def get_samples(self) -> tuple[list[list], list]:
        return pull_from_lsl_inlet(self.__inlet)

    def time_correction(self) -> float:
        return self.__inlet.time_correction()

    def __get_channel_properties(self, property: str) -> list[str]:
        properties = []
        descriptions = self.__info.desc().child("channels").child("channel")
        for i in range(self.nchannels):
            value = descriptions.child_value(property)
            properties.append(value)
            descriptions = descriptions.next_sibling()
        return properties


def pull_from_lsl_inlet(inlet: StreamInlet) -> tuple[list[list], list]:
    """StreamInlet.pull_chunk() may return None for samples.  This helper prevents None
    from propagating by converting it into [[]].  If None is detected, the timestamps list
    is also forced to [].
    """

    # read from inlet
    samples, timestamps = inlet.pull_chunk(timeout=0.1)

    # convert None into empty lists
    if samples is None:
        samples = [[]]
        timestamps = []

    # return tuple[list[list], list]
    return [samples, timestamps]
