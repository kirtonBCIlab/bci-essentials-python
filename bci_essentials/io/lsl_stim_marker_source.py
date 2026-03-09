from mne_lsl.lsl import StreamInlet, StreamInfo, resolve_streams

from .sources import MarkerSource
from ..triggers import DEFAULT_TRIGGER_MAP, TriggerDetector
from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)

__all__ = ["LslStimMarkerSource"]


class LslStimMarkerSource(MarkerSource):
    def __init__(
        self,
        stim_channel_name: str = "STIM",
        trigger_map: dict[int, str] | None = None,
        detect_mode: str = "rise",
        include_unmapped: bool = False,
        stream: StreamInfo | None = None,
        buffer_size: int = 5,
        timeout: float = 600,
    ):
        """Create a MarkerSource that reads triggers from the EEG stim channel.

        Opens its own LSL inlet to the EEG stream and scans the stim channel
        for trigger events, converting byte values to marker strings.

        Parameters
        ----------
        stim_channel_name : str, *optional*
            Label of the stim channel in the EEG stream. Default is "STIM".
        trigger_map : dict[int, str], *optional*
            Byte to marker string mapping. Defaults to DEFAULT_TRIGGER_MAP.
        detect_mode : str, *optional*
            "rise" (default) detects 0 to non-zero transitions.
            "change" detects any value change.
        include_unmapped : bool, *optional*
            If True, unmapped byte values produce "trigger_{value}" markers.
        stream : StreamInfo, *optional*
            Provide stream to use for EEG, if not provided, stream will be discovered.
        buffer_size : int, *optional*
            Size of the buffer to use for the inlet in seconds. Default is 5.
        timeout : float, *optional*
            How many seconds to wait for EEG stream to be discovered.
            If no stream is discovered, an Exception is raised.
            By default init will wait 10 minutes.
        """
        if detect_mode not in ("rise", "change"):
            raise ValueError(
                f"detect_mode must be 'rise' or 'change', got {detect_mode!r}"
            )

        self._stim_channel_name = stim_channel_name
        tmap = trigger_map if trigger_map is not None else dict(DEFAULT_TRIGGER_MAP)
        self._detector = TriggerDetector(tmap, detect_mode, include_unmapped)

        try:
            if stream is None:
                streams = resolve_streams(stype="EEG", timeout=timeout)
                if not streams:
                    raise RuntimeError(f"No EEG stream found within {timeout}s timeout")
                stream = streams[0]
            self._inlet = StreamInlet(
                stream, max_buffered=buffer_size, processing_flags=["dejitter"]
            )
            self._inlet.open_stream(timeout=5)
            self.__info = self._inlet.get_sinfo()
        except Exception:
            raise Exception(
                "LslStimMarkerSource: could not open LSL inlet to EEG stream"
            )

        self._stim_idx = self._find_stim_channel()

        self._samples_scanned: int = 0
        self._triggers_detected: int = 0
        self._warned_no_triggers: bool = False

    @property
    def name(self) -> str:
        return f"LslStimMarkerSource({self._stim_channel_name})"

    def get_markers(self) -> tuple[list[list], list]:
        samples, timestamps = self._inlet.pull_chunk(timeout=0.001)

        if samples is None or len(samples) == 0:
            return [[]], []

        markers = []
        marker_timestamps = []

        for sample, ts in zip(samples, timestamps):
            self._samples_scanned += 1

            if self._stim_idx is None or self._stim_idx >= len(sample):
                continue

            try:
                stim_val = int(sample[self._stim_idx])
            except (ValueError, TypeError):
                continue

            marker_str = self._detector.detect(stim_val)
            if marker_str is not None:
                self._triggers_detected += 1
                markers.append([marker_str])
                marker_timestamps.append(ts)
                logger.debug(f"trigger {stim_val} -> '{marker_str}' at t={ts:.4f}")

        # Warn once after ~10s of data with no triggers
        if (
            not self._warned_no_triggers
            and self._triggers_detected == 0
            and self._samples_scanned > 0
        ):
            try:
                warn_threshold = int(self.__info.sfreq * 10)
            except Exception:
                warn_threshold = 2560
            if self._samples_scanned >= warn_threshold:
                logger.warning(
                    f"Scanned {self._samples_scanned} samples from channel "
                    f"'{self._stim_channel_name}' with no triggers detected"
                )
                self._warned_no_triggers = True

        if not markers:
            return [[]], []

        return markers, marker_timestamps

    def time_correction(self) -> float:
        return self._inlet.time_correction()

    def _find_stim_channel(self) -> int | None:
        try:
            ch_names = self.__info.get_channel_names()
        except Exception:
            logger.warning("Could not read channel names from EEG stream")
            return None

        if self._stim_channel_name in ch_names:
            idx = ch_names.index(self._stim_channel_name)
            logger.info(f"Stim channel '{self._stim_channel_name}' at index {idx}")
            return idx

        logger.warning(
            f"Stim channel '{self._stim_channel_name}' not found in {ch_names}"
        )
        return None
