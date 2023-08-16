"""
**A package for online/offline processing of EEG-based BCIs.**

This library contains modules for the processing of EEG data for BCI
applications.

Supported applications are currently **P300 ERP** and **SSVEP**.

Modules are including for running **online** and **offline** processing
of EEG-based BCIs in an identical fashion.

Applications
------------
1. Load offline data
2. Stream online data
3. Provide options for visualizing BCI data

Limitations
-----------
1. Currently will not handle ERP sessions longer that 10,000 markers in
duration to avoid latency cause by dynamic sizing of numpy ndarrays.
This number can be increased by changing the `max_windows` variable in
the `ERP_data` class.

"""

from . import bci_data
from . import visuals
from . import signal_processing
from . import classification

__all__ = [
    "bci_data",
    "visuals",
    "classification",
    "signal_processing",
    "channel_selection",
    "resting_state",
]
