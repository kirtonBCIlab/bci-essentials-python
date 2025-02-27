"""
BCI Essentials Paradigm Submodule

This package defines BCI experiment paradigms:
- Base Paradigm: Base class defining common functionality for all paradigms
- MI Paradigm: Implements the Motor Imagery paradigm for motor movement/imagination tasks
- P300 Paradigm: Implements the P300/ERP paradigm for event-related potential detection
- SSVEP Paradigm: Implements the SSVEP paradigm for steady-state visual evoked potential detection
"""

# Import all paradigm modules to make them discoverable
from . import paradigm
from . import mi_paradigm
from . import p300_paradigm
from . import ssvep_paradigm