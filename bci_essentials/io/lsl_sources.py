from mne_lsl.lsl import StreamInlet, StreamInfo, resolve_streams
from .sources import MarkerSource, EegSource
from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)

__all__ = ["LslMarkerSource", "LslEegSource"]


class LslMarkerSource(MarkerSource):
    def __init__(
        self, stream: StreamInfo = None, buffer_size: int = 5, timeout: float = 600
    ):
        """Create a MarkerSource object that obtains markers from an LSL outlet

        Parameters
        ----------
        stream : StreamInfo, *optional*
            Provide stream to use for Markers, if not provided, stream will be discovered.
        buffer_size : int, *optional*
            Size of the buffer is `buffer_size * 100`. Default is 5 (i.e., 500 samples).
        timeout : float, *optional*
            How many seconds to wait for marker outlet stream to be discovered.
            If no stream is discovered, an Exception is raised.
            By default init will wait 10 minutes.
        """
        try:
            if stream is None:
                stream = discover_first_stream(
                    "BCI_Essentials_Markers", timeout=timeout
                )
            self._inlet = StreamInlet(
                stream, max_buffered=buffer_size, processing_flags=["dejitter"]
            )
            self._inlet.open_stream(timeout=5)
            self.__info = self._inlet.get_sinfo()
        except Exception:
            raise Exception("LslMarkerSource: could not create inlet")

    @property
    def name(self) -> str:
        return self.__info.name

    def get_markers(self) -> tuple[list[list], list]:
        return pull_from_lsl_inlet(self._inlet)

    def time_correction(self) -> float:
        return self._inlet.time_correction()


class LslEegSource(EegSource):
    def __init__(
        self, stream: StreamInfo = None, buffer_size: int = 5, timeout: float = 600
    ):
        """Create a MarkerSource object that obtains EEG from an LSL outlet

        Parameters
        ----------
        stream : StreamInfo, *optional*
            Provide stream to use for EEG, if not provided, stream will be discovered.
        buffer_size : int, *optional*
            Size of the buffer to use for the inlet in seconds. Default is 5.
        timeout : float, *optional*
            How many seconds to wait for marker outlet stream to be discovered.
            If no stream is discovered, an Exception is raised.
            By default init will wait 10 minutes.
        """
        try:
            if stream is None:
                stream = discover_first_stream("EEG", timeout=timeout)
            self._inlet = StreamInlet(
                stream, max_buffered=buffer_size, processing_flags=["dejitter"]
            )
            self._inlet.open_stream(timeout=5)
            self.__info = self._inlet.get_sinfo()
        except Exception:
            raise Exception("LslEegSource: could not create inlet")

    @property
    def name(self) -> str:
        return self.__info.name

    @property
    def fsample(self) -> float:
        return self.__info.sfreq

    @property
    def n_channels(self) -> int:
        return self.__info.n_channels

    @property
    def channel_types(self) -> list[str]:
        return self.__info.get_channel_types()

    @property
    def channel_units(self) -> list[str]:
        """Get channel units. Default to "microvolts"."""
        try:
            units = self.__info.get_channel_units()
            # If no units found or empty strings, use default
            if not units or all(unit == "" for unit in units):
                logger.warning("No channel units found, defaulting to microvolts")
                units = ["microvolts"] * self.n_channels
            return units
        except Exception:
            logger.warning("Could not get channel units, defaulting to microvolts")
            return ["microvolts"] * self.n_channels

    @property
    def channel_labels(self) -> list[str]:
        """Get channel labels.  Default to Ch1, Ch2, etc."""

        try:
            ch_names = self.__info.get_channel_names()
            # If no labels found or empty strings, use default
            if not ch_names or all(label == "" for label in ch_names):
                logger.warning("No channel labels found, defaulting to Ch1, Ch2, etc.")
                ch_names = [f"Ch{i + 1}" for i in range(self.n_channels)]
            return ch_names
        except Exception:
            logger.warning("Could not get channel labels, defaulting to Ch1, Ch2, etc.")
            return [f"Ch{i + 1}" for i in range(self.n_channels)]

        # if hasattr(self.__info, 'ch_names') and self.__info.ch_names:
        #     return list(self.__info.ch_names)
        # return [f"Ch{i+1}" for i in range(self.n_channels)]

    def get_samples(self) -> tuple[list[list], list]:
        return pull_from_lsl_inlet(self._inlet)

    def time_correction(self) -> float:
        return self._inlet.time_correction()

    def get_channel_properties(self, property: str) -> list[str]:
        """Get channel properties from mne_lsl stream info.

        Parameters
        ----------
        property : str
            Property to get ('name', 'unit', 'type', etc)

        Returns
        -------
        list[str]
            List of property values for each channel
        """
        if property == "name":
            return self.name
        elif property == "unit":
            return self.channel_units
        elif property == "type":
            return self.channel_types
        elif property == "label":
            return self.channel_labels
        else:
            logger.warning(f"Property '{property}' not supported in mne_lsl")
            return [""] * self.n_channels


def discover_first_stream(type: str, timeout: float = 600) -> StreamInfo:
    """This helper returns the first stream of the specified type.

    If no stream is found, an exception is raised."""
    streams = resolve_streams(stype=type, timeout=timeout)
    return streams[0]


def pull_from_lsl_inlet(inlet: StreamInlet) -> tuple[list[list], list]:
    """StreamInlet.pull_chunk() may return None for samples.

    This helper prevents `None` from propagating by converting it into [[]].

    If None is detected, the timestamps list is also forced to [].
    """

    # read from inlet
    samples, timestamps = inlet.pull_chunk(timeout=0.001)

    # convert None or empty samples into empty lists
    if samples is None or len(samples) == 0:
        samples = [[]]
        timestamps = []

    # return tuple[list[list], list]
    return [samples, timestamps]
