"""Example: Serial-port trigger ingestion — real and fake modes.

Demonstrates how to use the EEG stim-channel trigger pipeline with:

1. **Fake mode** — no hardware needed.  A ``FakeStimEegSource`` generates
   synthetic EEG data with trigger bytes injected on a schedule.  Uses
   ``EegStimTriggerMarkerSource`` (wrapper) since there is no LSL stream.

2. **Real mode** — uses ``LslStimMarkerSource`` with a live EEG stream
   whose last channel carries trigger codes from a hardware trigger box.

Usage::

    # Run in fake mode (default — no hardware required)
    python trigger_stim_example.py

    # Run in real mode (requires a live LSL EEG stream with stim channel)
    python trigger_stim_example.py --real
"""

import argparse
import time

from bci_essentials.triggers import make_simple_trigger_map
from bci_essentials.io.eeg_trigger_sources import EegStimTriggerMarkerSource
from bci_essentials.io.fake_stim_source import FakeStimEegSource
from bci_essentials.utils.logger import Logger

logger = Logger(name="trigger_stim_example")


# ======================================================================
# TRIGGER MAP — customise here
# ======================================================================
TRIGGER_MAP = make_simple_trigger_map(
    target_byte=10,
    non_target_byte=11,
    include_status=True,
)
# ======================================================================


def run_fake_mode() -> None:
    """Run the trigger pipeline with a FakeStimEegSource (no hardware)."""
    logger.info("=" * 60)
    logger.info("FAKE MODE — no hardware required")
    logger.info("=" * 60)

    schedule = [
        (0.0, 240),  # Trial Started
        (0.3, 10),  # TARGET flash
        (0.6, 11),  # non-target flash
        (0.9, 11),  # non-target flash
        (1.2, 10),  # TARGET flash
        (1.5, 241),  # Trial Ends
        (2.0, 243),  # Train Classifier
        (3.0, 240),  # Trial Started
        (3.3, 11),  # non-target
        (3.6, 11),  # non-target
        (3.9, 11),  # non-target
        (4.2, 241),  # Trial Ends
    ]

    fake_eeg = FakeStimEegSource(
        trigger_schedule=schedule,
        n_channels=9,  # 8 EEG + 1 stim
        stim_channel_index=-1,
        fsample=256.0,
        duration=5.0,
    )

    logger.info(f"Trigger map: {TRIGGER_MAP}")

    # Wrap in EegStimTriggerMarkerSource (for non-LSL sources)
    stim_source = EegStimTriggerMarkerSource(
        eeg_source=fake_eeg,
        stim_channel_name="STIM",
        trigger_map=TRIGGER_MAP,
        detect_mode="rise",
        include_unmapped=True,
    )

    logger.info("Reading samples and markers...")
    total_markers = 0

    for _ in range(100):
        samples, timestamps = stim_source.get_samples()
        if not timestamps or timestamps == []:
            if fake_eeg._samples_generated >= fake_eeg._total_samples:
                break
            time.sleep(0.05)
            continue

        markers, marker_ts = stim_source.get_markers()
        for marker, ts in zip(markers, marker_ts):
            total_markers += 1
            logger.info(f"  MARKER @ t={ts:.4f}s : {marker[0]}")

        time.sleep(0.05)

    logger.info(f"Done — detected {total_markers} trigger markers.")


def run_real_mode() -> None:
    """Run with a live LSL EEG stream using LslStimMarkerSource."""
    logger.info("=" * 60)
    logger.info("REAL MODE — requires live LSL EEG stream with stim channel")
    logger.info("=" * 60)

    try:
        from bci_essentials.io.lsl_stim_marker_source import LslStimMarkerSource
    except ImportError:
        logger.error("mne-lsl not installed — cannot use real mode.")
        return

    logger.info("Searching for LSL EEG stream (timeout 10 s)...")
    try:
        marker_source = LslStimMarkerSource(
            stim_channel_name="STIM",
            trigger_map=TRIGGER_MAP,
            detect_mode="rise",
            timeout=10,
        )
    except Exception as e:
        logger.error(f"Could not connect: {e}")
        return

    logger.info("Listening for triggers for 30 seconds...")
    deadline = time.monotonic() + 30
    total_markers = 0

    while time.monotonic() < deadline:
        markers, marker_ts = marker_source.get_markers()
        for marker, ts in zip(markers, marker_ts):
            total_markers += 1
            logger.info(f"  MARKER @ t={ts:.4f}s : {marker[0]}")
        time.sleep(0.01)

    logger.info(f"Done — detected {total_markers} trigger markers in 30 s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate serial-port trigger ingestion."
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use a live LSL EEG stream instead of the fake source.",
    )
    args = parser.parse_args()

    if args.real:
        run_real_mode()
    else:
        run_fake_mode()
