import pyxdf

from .sources import EegSource, MarkerSource


class XdfMarkerSource(MarkerSource):
    def __init__(self, filepath: str):
        samples, timestamps, info = load_xdf_stream(filepath, "LSL_Marker_Strings")
        self.__samples = samples
        self.__timestamps = timestamps
        self.__info = info

    @property
    def name(self) -> str:
        return self.__info["name"][0]

    def get_markers(self) -> tuple[list[list] | None, list]:
        # return all data on first get
        samples = self.__samples
        timestamps = self.__timestamps

        # reset to empty for next get
        self.__samples = None
        self.__timestamps = []

        return [samples, timestamps]

    def time_correction(self) -> float:
        return 0.0


class XdfEegSource(EegSource):
    def __init__(self, filepath: str):
        samples, timestamps, info = load_xdf_stream(filepath, "EEG")
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
    def nchannels(self) -> int:
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

    def get_samples(self) -> tuple[list[list] | None, list]:
        # return all data on first get
        samples = self.__samples
        timestamps = self.__timestamps

        # reset to empty for next get
        self.__samples = None
        self.__timestamps = []

        return [samples, timestamps]

    def time_correction(self) -> float:
        return 0.0


def load_xdf_stream(filepath: str, streamtype: str) -> tuple[list[list], list, list]:
    #  Initialize stream and info from xdf file if the streamtype is present
    #  Don't need the header returned by load_xdf()
    streams, _ = pyxdf.load_xdf(filepath)

    samples = [[]]
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
                print(streamtype + " data not available")
            break

    return (samples, timestamps, info)
