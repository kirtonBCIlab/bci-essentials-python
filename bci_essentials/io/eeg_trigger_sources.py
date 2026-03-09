from .sources import EegSource, MarkerSource
from ..triggers import TriggerDetector
from ..utils.logger import Logger  # Logger wrapper

# Re-export trigger map helpers for backward compatibility
from ..triggers import (  # noqa: F401
    DEFAULT_TRIGGER_MAP,
    SIMPLE_TRIGGER_MAP,
    make_p300_trigger_map,
    make_mi_trigger_map,
    make_ssvep_trigger_map,
    make_simple_trigger_map,
)
from ..triggers import TriggerByte as _TriggerByte  # noqa: F401

# Backward compat aliases
TARGET_BYTE: int = _TriggerByte.TARGET
NON_TARGET_BYTE: int = _TriggerByte.NON_TARGET

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)

__all__ = [
    "EegStimTriggerMarkerSource",
    # Re-exports from triggers.py (backward compat)
    "make_p300_trigger_map",
    "make_mi_trigger_map",
    "make_ssvep_trigger_map",
    "make_simple_trigger_map",
    "DEFAULT_TRIGGER_MAP",
    "SIMPLE_TRIGGER_MAP",
]


class EegStimTriggerMarkerSource(EegSource, MarkerSource):
    """Wraps an EegSource and extracts trigger events from its stim channel.

    Implements both EegSource and MarkerSource. EEG samples pass through
    unchanged; the stim channel is scanned for trigger events on each
    get_samples() call. For online LSL use, prefer LslStimMarkerSource.
    """

    def __init__(
        self,
        eeg_source: EegSource,
        stim_channel_name: str,
        trigger_map: dict[int, str] | None = None,
        detect_mode: str = "rise",
        include_unmapped: bool = False,
        enabled: bool = True,
    ):
        """Create an EegStimTriggerMarkerSource that wraps an EegSource.

        Parameters
        ----------
        eeg_source : EegSource
            Underlying EEG source (e.g. FakeStimEegSource).
        stim_channel_name : str
            Label of the stim channel in the EEG stream.
        trigger_map : dict[int, str], *optional*
            Byte to marker string mapping. Defaults to DEFAULT_TRIGGER_MAP.
        detect_mode : str, *optional*
            "rise" (default) or "change".
        include_unmapped : bool, *optional*
            If True, unmapped bytes produce "trigger_{value}".
        enabled : bool, *optional*
            Set False to skip trigger detection at runtime.
        """
        if not isinstance(eeg_source, EegSource):
            raise TypeError(
                f"eeg_source must be an EegSource, got {type(eeg_source).__name__}"
            )
        if not isinstance(stim_channel_name, str) or not stim_channel_name:
            raise ValueError(
                f"stim_channel_name must be a non-empty string, got {stim_channel_name!r}"
            )
        if detect_mode not in ("rise", "change"):
            raise ValueError(
                f"detect_mode must be 'rise' or 'change', got {detect_mode!r}"
            )

        if trigger_map is not None:
            for key in trigger_map:
                if not isinstance(key, int) or key < 0 or key > 255:
                    raise ValueError(
                        f"trigger_map keys must be integers in 0-255, got {key!r}"
                    )

        self._eeg = eeg_source
        self._stim_channel_name = stim_channel_name
        tmap = trigger_map if trigger_map is not None else dict(DEFAULT_TRIGGER_MAP)
        self._enabled = bool(enabled)
        self._detector = TriggerDetector(tmap, detect_mode, include_unmapped)

        self._pending: list[tuple[list[str], float]] = []
        self._resolved_stim_idx: int | None = None

        self._samples_scanned: int = 0
        self._triggers_detected: int = 0
        self._warned_no_triggers: bool = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    @property
    def stim_channel_name(self) -> str:
        return self._stim_channel_name

    @stim_channel_name.setter
    def stim_channel_name(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"stim_channel_name must be non-empty, got {name!r}")
        self._stim_channel_name = name
        self._resolved_stim_idx = None

    @property
    def name(self) -> str:
        return self._eeg.name

    @property
    def fsample(self) -> float:
        return self._eeg.fsample

    @property
    def n_channels(self) -> int:
        return self._eeg.n_channels

    @property
    def channel_types(self) -> list[str]:
        return self._eeg.channel_types

    @property
    def channel_units(self) -> list[str]:
        return self._eeg.channel_units

    @property
    def channel_labels(self) -> list[str]:
        return self._eeg.channel_labels

    def get_samples(self) -> tuple[list[list], list]:
        try:
            samples, timestamps = self._eeg.get_samples()
        except Exception:
            logger.error(
                "EegStimTriggerMarkerSource: error reading EEG source",
                exc_info=True,
            )
            return [[]], []

        if not self._enabled or not timestamps:
            return samples, timestamps

        stim_idx = self._resolve_stim_index()
        if stim_idx is None:
            return samples, timestamps

        for sample, ts in zip(samples, timestamps):
            if stim_idx >= len(sample):
                continue
            self._samples_scanned += 1
            try:
                stim_val = int(sample[stim_idx])
            except (ValueError, TypeError):
                continue

            marker_str = self._detector.detect(stim_val)
            if marker_str is not None:
                self._triggers_detected += 1
                self._pending.append(([marker_str], ts))

        self._check_no_trigger_warning()
        return samples, timestamps

    def time_correction(self) -> float:
        return self._eeg.time_correction()

    def get_markers(self) -> tuple[list[list], list]:
        if not self._enabled or not self._pending:
            return [[]], []
        markers = [m for m, _ in self._pending]
        timestamps = [t for _, t in self._pending]
        self._pending.clear()
        return markers, timestamps

    def _resolve_stim_index(self) -> int | None:
        if self._resolved_stim_idx is not None:
            return self._resolved_stim_idx
        try:
            labels = self._eeg.channel_labels
        except Exception:
            return None
        if self._stim_channel_name in labels:
            self._resolved_stim_idx = labels.index(self._stim_channel_name)
            return self._resolved_stim_idx
        logger.warning(
            f"Stim channel '{self._stim_channel_name}' not found in {labels}"
        )
        return None

    def _check_no_trigger_warning(self) -> None:
        if self._warned_no_triggers or self._triggers_detected > 0:
            return
        try:
            threshold = int(self._eeg.fsample * 10)
        except Exception:
            threshold = 2560
        if self._samples_scanned >= threshold:
            logger.warning(
                f"Scanned {self._samples_scanned} samples with no triggers. "
                f"Check stim_channel_name, trigger_map, and hardware connections."
            )
            self._warned_no_triggers = True
