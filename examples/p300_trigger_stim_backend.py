"""P300 BCI backend using the serial-port stim-channel trigger workflow.

Drop-in replacement for ``p300_unity_backend.py`` that reads BCI event markers
from the EEG stim channel instead of a separate LSL marker stream.

How it works
------------
1. Unity's ``SerialTriggerWriter`` sends a single byte over serial for every
   BCI event (stimulus flash, trial start/end, train classifier, etc.).
2. The hardware trigger box forwards the byte to the EEG amplifier, which
   embeds it in a dedicated stim channel alongside the EEG data.
3. ``LslEegSource`` receives the combined EEG+stim stream over LSL.
4. ``LslStimMarkerSource`` opens its own LSL inlet to the same EEG stream,
   reads rising edges on the stim channel, and converts them to marker
   strings using ``TRIGGER_MAP``.
5. ``BciController`` receives ``eeg_source`` and ``marker_source`` as
   separate objects — identical to the standard LSL marker workflow.
6. Predictions flow back to Unity over LSL via ``LslMessenger``.

Unity setup
-----------
On the BCIController GameObject:

1. Replace ``MarkerWriter`` with ``SerialCapableMarkerWriter``.
2. Add ``SerialTriggerWriter`` as a sibling component.
3. Set ``SerialTriggerWriter.PortName`` to your COM port (e.g. ``COM3``).
4. Leave ``UseSimpleTargetEncoding`` **disabled** (default) so that each
   stimulus flash is encoded as a unique byte (``stimulusIndex + 1``).

Configuration
-------------
Edit the constants in the ``CONFIGURATION`` section below, then run::

    python p300_trigger_stim_backend.py
"""

from bci_essentials.io.lsl_sources import LslEegSource
from bci_essentials.io.lsl_messenger import LslMessenger
from bci_essentials.io.lsl_stim_marker_source import LslStimMarkerSource
from bci_essentials.triggers import make_p300_trigger_map
from bci_essentials.bci_controller import BciController
from bci_essentials.paradigm.p300_paradigm import P300Paradigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.erp_rg_classifier import ErpRgClassifier

# ======================================================================
# CONFIGURATION — adjust these values to match your setup
# ======================================================================

# Label of the stim channel as it appears in your EEG LSL stream.
STIM_CHANNEL_NAME = "TRG"

# Number of P300 stimulus presenters (rows/columns or individual objects).
N_OPTIONS = 6

# ======================================================================

# Build the trigger map: status events (bytes 240-245) + one P300 entry
# per stimulus.  Default byte for stimulus i = i + 1.
TRIGGER_MAP = make_p300_trigger_map(
    n_options=N_OPTIONS,
)

# Connect to the live EEG stream (blocks until an LSL outlet appears).
eeg_source = LslEegSource()

# Marker source that reads triggers from the EEG stim channel.
# Opens its own LSL inlet — no dual-interface, no wrapping.
marker_source = LslStimMarkerSource(
    stim_channel_name=STIM_CHANNEL_NAME,
    trigger_map=TRIGGER_MAP,
)

# Outbound predictions still use LSL — no change from the standard backend.
messenger = LslMessenger()

# P300 paradigm + classifier (identical to p300_unity_backend.py)
paradigm = P300Paradigm()
data_tank = DataTank()
classifier = ErpRgClassifier()
classifier.set_p300_clf_settings(
    n_splits=5,
    lico_expansion_factor=1,
    oversample_ratio=0,
    undersample_ratio=0,
)

# Same pattern as the standard backend — separate eeg_source and marker_source.
controller = BciController(
    classifier, eeg_source, marker_source, paradigm, data_tank, messenger
)

controller.setup(online=True)
controller.run()
