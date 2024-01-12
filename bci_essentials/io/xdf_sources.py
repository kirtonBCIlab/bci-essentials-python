import pyxdf

from .sources import EegSource, MarkerSource
from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)

__all__ = ["XdfMarkerSource", "XdfEegSource"]


class XdfMarkerSource(MarkerSource):
    def __init__(self, filename: str):
        """Create a MarkerSource object that obtains markers from an XDF file.

        Parameters
        ----------
        filename : str
            The full name of file, including path.  If file isn't found, an Exception is raised.
        """
        samples, timestamps, info = load_xdf_stream(filename, "LSL_Marker_Strings")
        self.__samples = samples
        self.__timestamps = timestamps
        self.__info = info

    @property
    def name(self) -> str:
        return self.__info["name"][0]

    def get_markers(self) -> tuple[list[list], list]:
        """Read markers and related timestamps from the XDF file.  Returns the contents
        of the file on the first call to get_markers(), returns empty lists thereafter.
        """
        # return all data on first get
        samples = self.__samples
        timestamps = self.__timestamps

        # reset to empty for next get
        self.__samples = [[]]
        self.__timestamps = []

        return [samples, timestamps]

    def time_correction(self) -> float:
        return 0.0


class XdfEegSource(EegSource):
    """Create a MarkerSource object that obtains EEG from an XDF file

    Parameters
    ----------
    filename : str
        The full name of file, including path. If file isn't found, an Exception is raised.
    """

    def __init__(self, filename: str):
        samples, timestamps, info = load_xdf_stream(filename, "EEG")
        self.__samples = samples
        self.__timestamps = timestamps
        self.__info = info

    @property
    def name(self) -> str:
        return self.__info["name"][0]

    @property
    def fsample(self) -> float:
        return float(self.__info["nominal_srate"][0])

    @property
    def n_channels(self) -> int:
        return int(self.__info["channel_count"][0])

    @property
    def channel_types(self) -> list[str]:
        types = []
        try:
            description = self.__info["desc"][0]["channels"][0]["channel"]
            types = [channel["type"][0] for channel in description]
            types = [str.lower(type) for type in types]
        except Exception:
            pass

        return types

    @property
    def channel_units(self) -> list[str]:
        units = []
        try:
            description = self.__info["desc"][0]["channels"][0]["channel"]
            units = [channel["unit"][0] for channel in description]
        except Exception:
            pass
        return units

    @property
    def channel_labels(self) -> list[str]:
        labels = []
        try:
            description = self.__info["desc"][0]["channels"][0]["channel"]
            labels = [channel["label"][0] for channel in description]
        except Exception:
            pass
        return labels

    def get_samples(self) -> tuple[list[list], list]:
        """Read markers and related timestamps from the XDF file.  Returns the contents of the file
        on the first call to get_markers(), returns empty lists thereafter.
        """
        # return all data on first get
        samples = self.__samples
        timestamps = self.__timestamps

        # reset to empty for next get
        self.__samples = [[]]
        self.__timestamps = []

        return [samples, timestamps]

    def time_correction(self) -> float:
        return 0.0


def load_xdf_stream(filepath: str, streamtype: str) -> tuple[list, list, list]:
    """A helper function to load the contents of the XDF file and return stream data, timestamps and info"""
    #  Don't need the header returned by load_xdf()
    streams, _ = pyxdf.load_xdf(filepath)

    samples = []
    timestamps = []
    info = []

    for i in range(len(streams)):
        stream = streams[i]
        info = stream["info"]
        type = info["type"][0]
        if type == streamtype:
            try:
                samples = stream["time_series"]
                timestamps = stream["time_stamps"]

            except Exception:
                logger.error("%s data not available", streamtype)
            break

    return (samples, timestamps, info)
