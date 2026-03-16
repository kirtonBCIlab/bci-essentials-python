from enum import Enum

__all__ = [
    "MarkerTypes",
    "TriggerByte",
    "TriggerDetector",
    "DEFAULT_TRIGGER_MAP",
    "SIMPLE_TRIGGER_MAP",
    "make_p300_trigger_map",
    "make_mi_trigger_map",
    "make_ssvep_trigger_map",
    "make_simple_trigger_map",
]


class MarkerTypes(Enum):
    """Status marker strings recognised by BciController.

    These correspond 1-to-1 with the status bytes in TriggerByte.
    """

    TRIAL_STARTED = "Trial Started"
    TRIAL_ENDS = "Trial Ends"
    TRAINING_COMPLETE = "Training Complete"
    TRAIN_CLASSIFIER = "Train Classifier"
    UPDATE_CLASSIFIER = "Update Classifier"
    DONE_RS_COLLECTION = "Done with all RS collection"

    # Resting state markers
    START_EYES_OPEN = "Start Eyes Open RS: 1"
    END_EYES_OPEN = "End Eyes Open RS: 1"
    START_EYES_CLOSED = "Start Eyes Closed RS: 2"
    END_EYES_CLOSED = "End Eyes Closed RS: 2"
    START_REST = "Start Rest for RS: 0"
    END_REST = "End Rest for RS: 0"


class TriggerByte:
    """Default serial-port byte values sent by Unity's SerialTriggerWriter.

    Status bytes occupy the high range (240-255) to stay clear of
    stimulus byte values (1-N).
    """

    # Status events
    TRIAL_STARTED: int = 240
    TRIAL_ENDS: int = 241
    TRAINING_COMPLETE: int = 242
    TRAIN_CLASSIFIER: int = 243
    UPDATE_CLASSIFIER: int = 244
    DONE_RS_COLLECTION: int = 245

    # Simple target/non-target encoding
    TARGET: int = 1
    NON_TARGET: int = 2


# Default trigger map for the six status events (byte -> marker string)
DEFAULT_TRIGGER_MAP: dict[int, str] = {
    TriggerByte.TRIAL_STARTED: MarkerTypes.TRIAL_STARTED.value,
    TriggerByte.TRIAL_ENDS: MarkerTypes.TRIAL_ENDS.value,
    TriggerByte.TRAINING_COMPLETE: MarkerTypes.TRAINING_COMPLETE.value,
    TriggerByte.TRAIN_CLASSIFIER: MarkerTypes.TRAIN_CLASSIFIER.value,
    TriggerByte.UPDATE_CLASSIFIER: MarkerTypes.UPDATE_CLASSIFIER.value,
    TriggerByte.DONE_RS_COLLECTION: MarkerTypes.DONE_RS_COLLECTION.value,
}

# Simplified target/non-target trigger map
SIMPLE_TRIGGER_MAP: dict[int, str] = {
    TriggerByte.TARGET: "target",
    TriggerByte.NON_TARGET: "non_target",
}


def make_simple_trigger_map(
    target_byte: int = TriggerByte.TARGET,
    non_target_byte: int = TriggerByte.NON_TARGET,
    include_status: bool = True,
    custom: dict[int, str] | None = None,
) -> dict[int, str]:
    """Build a simplified target / non-target trigger map.

    Parameters
    ----------
    target_byte : int
        Byte value for the target stimulus (default 1).
    non_target_byte : int
        Byte value for non-target stimuli (default 2).
    include_status : bool
        If True (default), merge DEFAULT_TRIGGER_MAP into the returned dict.
    custom : dict[int, str] or None
        Extra entries to merge. Overrides colliding keys.

    Returns
    -------
    dict[int, str]
    """
    if not (0 <= target_byte <= 255):
        raise ValueError(f"target_byte must be 0-255, got {target_byte}")
    if not (0 <= non_target_byte <= 255):
        raise ValueError(f"non_target_byte must be 0-255, got {non_target_byte}")
    if target_byte == non_target_byte:
        raise ValueError(
            f"target_byte and non_target_byte must differ, both are {target_byte}"
        )

    trigger_map: dict[int, str] = {}
    if include_status:
        trigger_map.update(DEFAULT_TRIGGER_MAP)
    trigger_map[target_byte] = "target"
    trigger_map[non_target_byte] = "non_target"
    if custom:
        trigger_map.update(custom)
    return trigger_map


def make_p300_trigger_map(
    n_options: int,
    stimulus_base_byte: int = 1,
) -> dict[int, str]:
    """Build a trigger map for a single-flash P300 paradigm.

    Unity sends byte = StimulusIndex + 1 for each flash. The byte encodes
    which object flashed, not which object is the training target. All markers
    use -1 (classification mode) since the target changes per trial.

    Marker format: "p300,s,{n_options},-1,{flash_index}"

    Parameters
    ----------
    n_options : int
        Number of stimulus presenters.
    stimulus_base_byte : int
        First byte value for stimuli. Default 1.

    Returns
    -------
    dict[int, str]
    """
    if n_options <= 0:
        raise ValueError(f"n_options must be positive, got {n_options}")
    if n_options > 239:
        raise ValueError(f"n_options must be at most 239, got {n_options}")

    trigger_map = dict(DEFAULT_TRIGGER_MAP)
    for i in range(n_options):
        byte_code = stimulus_base_byte + i
        trigger_map[byte_code] = f"p300,s,{n_options},-1,{i + 1}"
    return trigger_map


def make_mi_trigger_map(
    n_classes: int,
    epoch_length: float,
) -> dict[int, str]:
    """Build a trigger map for a Motor Imagery paradigm.

    Unity sends byte = TrainingTargetIndex + 1 for each MI trial.
    Each byte maps to its own class label (1-indexed).

    Marker format: "mi,{n_classes},{label},{epoch_length}"

    Parameters
    ----------
    n_classes : int
        Number of MI classes.
    epoch_length : float
        EEG epoch length in seconds.

    Returns
    -------
    dict[int, str]
    """
    if n_classes <= 0:
        raise ValueError(f"n_classes must be positive, got {n_classes}")
    if epoch_length <= 0:
        raise ValueError(f"epoch_length must be positive, got {epoch_length}")
    if n_classes > 239:
        raise ValueError(f"n_classes must be at most 239, got {n_classes}")

    trigger_map = dict(DEFAULT_TRIGGER_MAP)
    for i in range(n_classes):
        byte_code = i + 1
        trigger_map[byte_code] = f"mi,{n_classes},{i + 1},{epoch_length:.2f}"
    return trigger_map


def make_ssvep_trigger_map(
    frequencies: list[float],
    epoch_length: float,
) -> dict[int, str]:
    """Build a trigger map for an SSVEP paradigm.

    Unity sends byte = TrainingTargetIndex + 1 for each SSVEP trial.
    Each byte maps to its own class label (1-indexed).

    Marker format: "ssvep,{n_frequencies},{label},{epoch_length},{freq1},{freq2},..."

    Parameters
    ----------
    frequencies : list[float]
        Flashing frequencies for stimulus presenters.
    epoch_length : float
        EEG epoch length in seconds.

    Returns
    -------
    dict[int, str]
    """
    n_freqs = len(frequencies)
    if n_freqs == 0:
        raise ValueError("frequencies must not be empty")
    if epoch_length <= 0:
        raise ValueError(f"epoch_length must be positive, got {epoch_length}")
    if n_freqs > 239:
        raise ValueError(f"number of frequencies must be at most 239, got {n_freqs}")

    freq_str = ",".join(str(f) for f in frequencies)
    trigger_map = dict(DEFAULT_TRIGGER_MAP)
    for i in range(n_freqs):
        byte_code = i + 1
        trigger_map[byte_code] = (
            f"ssvep,{n_freqs},{i + 1}," f"{epoch_length:.2f},{freq_str}"
        )
    return trigger_map


class TriggerDetector:
    """Stateful trigger detector for stim-channel values.

    Encapsulates the rise/change detection logic used by
    LslStimMarkerSource and EegStimTriggerMarkerSource.
    """

    def __init__(
        self,
        trigger_map: dict[int, str],
        detect_mode: str = "rise",
        include_unmapped: bool = False,
    ):
        """Create a TriggerDetector.

        Parameters
        ----------
        trigger_map : dict[int, str]
            Byte to marker string mapping.
        detect_mode : str, *optional*
            "rise" or "change". Default is "rise".
        include_unmapped : bool, *optional*
            If True, unmapped bytes produce "trigger_{value}".
        """
        if detect_mode not in ("rise", "change"):
            raise ValueError(
                f"detect_mode must be 'rise' or 'change', got {detect_mode!r}"
            )
        self.trigger_map = trigger_map
        self.detect_mode = detect_mode
        self.include_unmapped = include_unmapped
        self.last_value: int = 0
        self.warned_unmapped: set[int] = set()

    def detect(self, stim_val: int) -> str | None:
        """Process a stim-channel sample value. Returns the marker string
        if a trigger event is detected, otherwise None."""
        is_event = False
        if self.detect_mode == "rise":
            is_event = stim_val != 0 and self.last_value == 0
        else:  # "change"
            is_event = stim_val != self.last_value

        self.last_value = stim_val

        if not is_event or stim_val == 0:
            return None

        if stim_val in self.trigger_map:
            return self.trigger_map[stim_val]
        elif self.include_unmapped:
            return f"trigger_{stim_val}"
        else:
            self.warned_unmapped.add(stim_val)
            return None
