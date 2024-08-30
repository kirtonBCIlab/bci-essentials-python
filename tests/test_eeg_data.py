import unittest

import numpy as np
import time

from bci_essentials.io.sources import MarkerSource, EegSource
from bci_essentials.io.messenger import Messenger

# from bci_essentials.paradigm.paradigm import Paradigm
from bci_essentials.paradigm.mi_paradigm import MiParadigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.bci_controller import BciController
from bci_essentials.classification.generic_classifier import Prediction
from bci_essentials.classification.null_classifier import NullClassifier


class TestBciController(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = NullClassifier()
        self.eeg = _MockEegSource()
        self.markers = _MockMarkerSource()
        self.messenger = _MockMessenger()

    def test_trg_channel_types_changed_to_stim(self):
        self.eeg.channel_types = ["eeg", "trg", "eeg", "stim"]
        data = BciController(
            self.classifier, self.eeg, self.markers, MiParadigm(), DataTank()
        )
        self.assertEqual(data.ch_type, ["eeg", "stim", "eeg", "stim"])

    # offline
    def test_when_offline_loop_stops_when_no_more_data(self):
        data = BciController(
            self.classifier,
            self.eeg,
            self.markers,
            MiParadigm(),
            DataTank(),
            self.messenger,
        )
        data.setup(online=False)
        data.run(max_loops=200)
        self.assertEqual(self.messenger.ping_count, 1)

    def test_offline_single_step_runs_single_loop(self):
        data = BciController(
            self.classifier,
            self.eeg,
            self.markers,
            MiParadigm(),
            DataTank(),
            self.messenger,
        )
        data.setup(online=False)

        data.step()
        self.assertEqual(self.messenger.ping_count, 1)
        data.step()
        self.assertEqual(self.messenger.ping_count, 2)

    # online
    def test_when_online_loop_continues_even_when_no_data(self):
        data = BciController(
            self.classifier,
            self.eeg,
            self.markers,
            MiParadigm(),
            DataTank(),
            self.messenger,
        )
        data.setup(online=True)
        data.run(max_loops=10)
        self.assertEqual(self.messenger.ping_count, 10)

    def test_online_single_step_runs_single_loop(self):
        data = BciController(
            self.classifier,
            self.eeg,
            self.markers,
            MiParadigm(),
            DataTank(),
            self.messenger,
        )
        data.setup(online=False)

        data.step()
        self.assertEqual(self.messenger.ping_count, 1)
        data.step()
        self.assertEqual(self.messenger.ping_count, 2)

    def test_step_does_not_wait_for_more_data(self):
        data_tank = DataTank()

        data = BciController(
            self.classifier,
            self.eeg,
            self.markers,
            MiParadigm(),
            data_tank,
            self.messenger,
        )
        data.setup(online=True)
        data.step()

        # Add some fake EEG data
        data_tank.add_raw_eeg(np.array([1, 2, 3]), np.array([0, 1, 2]))

        # Add a marker for which we have no data
        data_tank.add_raw_markers(np.array(["mi,2,-1,0.5"]), np.array([100]))

        # Run another step, it should not wait for more data (< 1s)
        start = time.time()
        data.step()
        end = time.time()

        self.assertLess(end - start, 1)

    def test_when_online_and_invalid_markers_then_loop_continues(self):
        data = BciController(
            self.classifier,
            self.eeg,
            self.markers,
            MiParadigm(),
            DataTank(),
            self.messenger,
        )

        # provide garbage data (None is invalid, length of data and timestamps doesn't match)
        self.markers.marker_data = [1.0]
        self.markers.marker_timestamps = [1.0]
        data.setup(online=True)
        data.run(max_loops=2)

        # if we didn't crash, sanity check that loops did happen
        self.assertEqual(self.messenger.ping_count, 2)

    def test_when_online_and_invalid_eeg_then_loop_continues(self):
        data = BciController(
            self.classifier,
            self.eeg,
            self.markers,
            MiParadigm(),
            DataTank(),
            self.messenger,
        )

        # provide garbage data (None is invalid, length of data and timestamps doesn't match)
        self.eeg.bci_controller = [1.0]
        self.eeg.eeg_timestamps = [1.0]
        data.setup(online=True)
        data.run(max_loops=2)

        # if we didn't crash, sanity check that loops did happen
        self.assertEqual(self.messenger.ping_count, 2)


# Placeholder to make BciController happy
class _MockMarkerSource(MarkerSource):
    name = "MockMarker"
    marker_data = [[]]
    marker_timestamps = []

    def get_markers(self) -> tuple[list[list], list]:
        return [self.marker_data, self.marker_timestamps]

    def time_correction(self) -> float:
        return 0.0


# Placeholder to make BciController happy
class _MockEegSource(EegSource):
    name = "MockEeg"
    fsample = 0.0
    n_channels = 0
    channel_types = []
    channel_units = []
    channel_labels = []
    bci_controller = [[]]
    eeg_timestamps = []

    def get_samples(self) -> tuple[list[list], list]:
        return [self.bci_controller, self.eeg_timestamps]

    def time_correction(self) -> float:
        return 0.0


# Count how often these methods are called to check that BciController
# would have sent a message, not checking content.
class _MockMessenger(Messenger):
    ping_count = 0
    marker_received_count = 0
    prediction_count = 0

    def ping(self):
        self.ping_count += 1

    def marker_received(self, marker):
        self.marker_received_count += 1

    def prediction(self, prediction: Prediction):
        self.prediction_count += 1
