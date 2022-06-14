"""
A package for online/offline processing of EEG-based BCIs


"""

from . import bci_data
from . import visuals
from . import classification
from . import signal_processing

__all__ = ["bci_data", "visuals", "classification", "signal_processing"]