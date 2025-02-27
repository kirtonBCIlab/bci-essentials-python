"""
BCI Essentials
==============

A collection of tools and methods for Brain-Computer Interface (BCI) development, 
analysis, and implementation.

Root Level Modules:
------------------
- bci_controller: Contains the main BciController class that manages EEG data processing, 
  trial management, and classification
- signal_processing: Provides functions for filtering (bandpass, lowpass, highpass, notch) 
  and signal processing
- channel_selection: Implements algorithms for selecting optimal channels for BCI performance
- session_saving: Tools for saving and loading classifier models and sessions
- resting_state: Functions for analyzing resting state EEG data, including bandpower 
  features and alpha peak detection

Submodules:
----------
- classification: Various classifier implementations for different BCI paradigms
- data_tank: Storage management for EEG data
- io: Handles data acquisition and communication
- paradigm: Defines BCI experiment paradigms
- utils: Utility functions used throughout the library
"""

# Import submodules to make them discoverable
from . import classification
from . import data_tank
from . import io
from . import paradigm
from . import utils

# Import root-level modules to ensure they're documented by pdoc
from . import bci_controller
from . import signal_processing
from . import channel_selection
from . import session_saving
from . import resting_state

# Instantiate a parent logger for the library at the default level of logging.INFO
from .utils.logger import Logger

logger = Logger()