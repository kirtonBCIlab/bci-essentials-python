import time

from .sources import EegSource
from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)

__all__ = ["FakeStimEegSource"]


class FakeStimEegSource(EegSource):
    """Synthetic EEG source with programmable stim-channel trigger injection.

    Generates zero-valued EEG samples at the configured sample rate. One channel
    (the stim channel) carries trigger byte values injected at regular intervals
    or according to a user-supplied schedule. Designed for testing without hardware.
    """

    def __init__(
        self,
        n_channels: int = 9,
        fsample: float = 256.0,
        stim_channel_index: int = -1,
        trigger_bytes: list[int] | None = None,
        trigger_interval: float = 1.0,
        trigger_schedule: list[tuple[float, int]] | None = None,
        duration: float = 10.0,
        channel_labels: list[str] | None = None,
    ):
        """Create a FakeStimEegSource for testing.

        Parameters
        ----------
        n_channels : int, *optional*
            Total number of channels (EEG + stim). Default is 9.
        fsample : float, *optional*
            Sample rate in Hz. Default is 256.0.
        stim_channel_index : int, *optional*
            Index of the stim channel. Default is -1 (last channel).
        trigger_bytes : list[int], *optional*
            Byte values cycled through at trigger_interval spacing.
            Mutually exclusive with trigger_schedule.
        trigger_interval : float, *optional*
            Seconds between successive triggers when using trigger_bytes.
            Default is 1.0.
        trigger_schedule : list[tuple[float, int]], *optional*
            Explicit (time_offset_sec, byte_value) pairs.
            Mutually exclusive with trigger_bytes.
        duration : float, *optional*
            Total duration of fake data in seconds. Default is 10.0.
        channel_labels : list[str], *optional*
            Custom channel labels. Defaults to ["Ch0", "Ch1", ..., "STIM"].
        """
        if trigger_bytes is not None and trigger_schedule is not None:
            raise ValueError(
                "Specify either trigger_bytes or trigger_schedule, not both."
            )

        self._n_channels = n_channels
        self._fsample = fsample
        self._duration = duration

        # Resolve stim channel index
        self._stim_idx = (
            stim_channel_index
            if stim_channel_index >= 0
            else n_channels + stim_channel_index
        )

        # Channel labels
        if channel_labels is not None:
            if len(channel_labels) != n_channels:
                raise ValueError(
                    f"channel_labels length ({len(channel_labels)}) "
                    f"!= n_channels ({n_channels})"
                )
            self._channel_labels = list(channel_labels)
        else:
            self._channel_labels = [
                f"Ch{i}" if i != self._stim_idx else "STIM" for i in range(n_channels)
            ]

        # Build the trigger schedule (sorted by time)
        if trigger_schedule is not None:
            self._schedule = sorted(trigger_schedule, key=lambda x: x[0])
        elif trigger_bytes is not None:
            self._schedule = []
            t = 0.0
            idx = 0
            while t < duration:
                self._schedule.append((t, trigger_bytes[idx % len(trigger_bytes)]))
                t += trigger_interval
                idx += 1
        else:
            self._schedule = []

        self._start_time: float | None = None
        self._samples_generated: int = 0
        self._total_samples = int(duration * fsample)
        self._schedule_idx = 0

    @property
    def name(self) -> str:
        return "FakeStimEegSource"

    @property
    def fsample(self) -> float:
        return self._fsample

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def channel_types(self) -> list[str]:
        return [
            "stim" if i == self._stim_idx else "eeg" for i in range(self._n_channels)
        ]

    @property
    def channel_units(self) -> list[str]:
        return [
            "n/a" if i == self._stim_idx else "microvolts"
            for i in range(self._n_channels)
        ]

    @property
    def channel_labels(self) -> list[str]:
        return self._channel_labels

    def get_samples(self) -> tuple[list[list], list]:
        if self._samples_generated >= self._total_samples:
            return [[]], []

        now = time.monotonic()
        if self._start_time is None:
            self._start_time = now

        elapsed = now - self._start_time
        target_sample = min(int(elapsed * self._fsample), self._total_samples)
        n_new = target_sample - self._samples_generated

        if n_new <= 0:
            return [[]], []

        samples = []
        timestamps = []

        for i in range(n_new):
            sample_idx = self._samples_generated + i
            t = self._start_time + sample_idx / self._fsample

            # Determine stim value for this sample
            stim_val = 0
            sample_time_offset = sample_idx / self._fsample
            while (
                self._schedule_idx < len(self._schedule)
                and self._schedule[self._schedule_idx][0] <= sample_time_offset
            ):
                stim_val = self._schedule[self._schedule_idx][1]
                self._schedule_idx += 1

            row = [0.0] * self._n_channels
            row[self._stim_idx] = float(stim_val)
            samples.append(row)
            timestamps.append(t)

        self._samples_generated = target_sample
        return samples, timestamps

    def time_correction(self) -> float:
        return 0.0
