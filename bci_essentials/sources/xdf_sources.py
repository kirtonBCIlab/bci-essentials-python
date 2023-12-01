import pyxdf

from .sample_source import SampleSource


class XdfSampleSource(SampleSource):
    def __init__(self, filepath: str, streamtype: str):
        self.__info = None
        self.__samples: list[list] = [[]]
        self.__timestamps: list = []
        self.__load_xdf_stream(filepath, streamtype)

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
        for channel in range(self.nchannels):
            try:
                description = self.__info["desc"][0]["channels"][0]["channel"][channel]
                type = description["type"][0]
                type = type.lower()

                # save trigger type channel as stim
                if type == "trg":
                    type = "stim"

                types.append(type)
            except Exception:
                break

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
        for channel in range(self.nchannels):
            try:
                description = self.__info["desc"][0]["channels"][0]["channel"][channel]
                label = description["label"][0]
            except Exception:
                label = "?"
            labels.append(label)

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

    def __load_xdf_stream(self, filepath: str, streamtype: str):
        #  Initialize stream and info from xdf file if the streamtype is present
        #  Don't need the header returned by load_xdf()
        streams, _ = pyxdf.load_xdf(filepath)
        for i in range(len(streams)):
            stream = streams[i]
            info = stream["info"]
            type = info["type"][0]
            if type == streamtype:
                try:
                    self.__info = info
                    self.__samples = stream["time_series"]
                    self.__timestamps = stream["time_stamps"]

                except Exception:
                    print(streamtype + " data not available")
                break


class XdfMarkerSource(XdfSampleSource):
    def __init__(self, filepath: str):
        super().__init__(filepath, "LSL_Marker_Strings")


class XdfEegSource(XdfSampleSource):
    def __init__(self, filepath: str):
        super().__init__(filepath, "EEG")
