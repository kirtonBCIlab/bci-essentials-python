"""BCI Essentials I/O — data sources and messengers.

Quick imports::

    from bci_essentials.io import LslEegSource, LslMarkerSource, LslMessenger
    from bci_essentials.io import XdfEegSource, XdfMarkerSource
    from bci_essentials.io import LslStimMarkerSource
"""

# Abstract base classes
from .sources import EegSource, MarkerSource
from .messenger import Messenger

# LSL (online)
from .lsl_sources import LslEegSource, LslMarkerSource
from .lsl_messenger import LslMessenger
from .lsl_stim_marker_source import LslStimMarkerSource

# XDF (offline)
from .xdf_sources import XdfEegSource, XdfMarkerSource

# Testing
from .fake_stim_source import FakeStimEegSource

# Wrapper (testing / offline stim-channel extraction)
from .eeg_trigger_sources import EegStimTriggerMarkerSource

__all__ = [
    # ABCs
    "EegSource",
    "MarkerSource",
    "Messenger",
    # LSL
    "LslEegSource",
    "LslMarkerSource",
    "LslMessenger",
    "LslStimMarkerSource",
    # XDF
    "XdfEegSource",
    "XdfMarkerSource",
    # Testing
    "FakeStimEegSource",
    "EegStimTriggerMarkerSource",
]
