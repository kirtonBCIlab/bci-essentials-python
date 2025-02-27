"""
BCI Essentials IO Submodule

This package handles data acquisition and communication:
- Sources: Abstract base classes defining interfaces for EEG and marker data sources
- Messenger: Interface for communication between BCI components and external applications
- LSL Sources: Lab Streaming Layer (LSL) implementations of EEG and marker sources
- LSL Messenger: LSL implementation of the messenger interface for sending events
- XDF Sources: XDF file-based implementations of EEG and marker sources for offline analysis
"""

# Import all IO modules to make them discoverable
from . import sources
from . import messenger
from . import lsl_sources
from . import lsl_messenger
from . import xdf_sources