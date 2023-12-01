from abc import ABC, abstractmethod


class MarkerSource(ABC):
    """
    MarkerSource objects send time synchronized markers/commands to EEG_data.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_markers(self) -> tuple[list, list]:
        pass

    @abstractmethod
    def time_correction(self) -> float:
        pass


class EegSource(ABC):
    """
    EegSource objects produce samples of EEG for use in EEG_data.

    It can be used to represent an BCI headset providing EEG data, or it could be a source
    of Markers to control EEG_data behaviour, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def fsample(self) -> float:
        pass

    @property
    @abstractmethod
    def nchannels(self) -> int:
        pass

    @property
    @abstractmethod
    def channel_types(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def channel_units(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def channel_labels(self) -> list[str]:
        pass

    @abstractmethod
    def get_samples(self) -> tuple[list, list]:
        pass

    @abstractmethod
    def time_correction(self) -> float:
        pass
