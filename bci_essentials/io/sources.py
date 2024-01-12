from abc import ABC, abstractmethod

from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


class MarkerSource(ABC):
    """MarkerSource objects send time synchronized markers/commands to EEG_data."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the marker source"""
        pass

    @abstractmethod
    def get_markers(self) -> tuple[list[list], list]:
        """Get marker/command data and timestamps since last call

        Returns
        -------
            A tuple of (markers, timestamps):

            markers - A list of samples, where each sample corresponds to a timestamp.
            Each sample is a list with a single string element that represents a command or
            a marker.  The string is formatted as follows:
            command = an arbitrary string, ex: "Trial Started".
            marker = "paradigm, num options, label number, trial length"

            timestamps - A list timestamps (float in seconds) corresponding to the markers
        """
        pass

    @abstractmethod
    def time_correction(self) -> float:
        """Get the current time correction for timestamps.

        Returns
        -------
            The current time correction estimate (float, seconds). This is the number that needs to be added
            to a time tamp that was remotely generated via local_clock() to map it into
            the local clock domain of the machine.
        """
        pass


class EegSource(ABC):
    """EegSource objects produce samples of EEG for use in EEG_data.

    It can be used to represent an BCI headset providing EEG data, or it could be a source
    of Markers to control EEG_data behaviour, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the EEG source"""
        pass

    @property
    @abstractmethod
    def fsample(self) -> float:
        """Sample rate of EEG source"""
        pass

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of EEG channels per sample"""
        pass

    @property
    @abstractmethod
    def channel_types(self) -> list[str]:
        """The type of each channel, ex: eeg, or stim"""
        pass

    @property
    @abstractmethod
    def channel_units(self) -> list[str]:
        """The unit of each channel, ex: microvolts"""
        pass

    @property
    @abstractmethod
    def channel_labels(self) -> list[str]:
        """The label for each channel, ex: FC3, C5"""
        pass

    @abstractmethod
    def get_samples(self) -> tuple[list[list], list]:
        """Get EEG samples and timestamps since last call

        Returns
        -------
            A tuple of (samples, timestamps):

            samples - A list of samples, where each sample corresponds to a timestamp.
            Each sample is a list of floats representing the value for each channel of EEG.

            timestamps - A list timestamps (float in seconds) corresponding to the samples
        """
        pass

    @abstractmethod
    def time_correction(self) -> float:
        """Get the current time correction for timestamps.

        Returns
        -------
            The current time correction estimate (float, seconds). This is the number that needs to be added
            to a time tamp that was remotely generated via local_clock() to map it into
            the local clock domain of the machine.
        """
        pass
