import unittest

from bci_essentials.io.sources import MarkerSource, EegSource
from bci_essentials.io.messenger import Messenger
from bci_essentials.eeg_data import EegData
from bci_essentials.classification.generic_classifier import Prediction
from bci_essentials.classification.null_classifier import NullClassifier


class TestEegData(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = NullClassifier()
        self.eeg = _MockEegSource()
        self.markers = _MockMarkerSource()
        self.messenger = _MockMessenger()

    def test_trg_channel_types_changed_to_stim(self):
        self.eeg.channel_types = ["eeg", "trg", "eeg", "stim"]
        data = EegData(self.classifier, self.eeg, self.markers)
        self.assertEqual(data.ch_type, ["eeg", "stim", "eeg", "stim"])

    # offline
    def test_when_offline_loop_stops_when_no_more_data(self):
        data = EegData(self.classifier, self.eeg, self.markers, self.messenger)
        data.setup(online=False)
        data.run(max_loops=200)
        self.assertEqual(self.messenger.ping_count, 1)

    def test_offline_single_step_runs_single_loop(self):
        data = EegData(self.classifier, self.eeg, self.markers, self.messenger)
        data.setup(online=False)

        data.step()
        self.assertEqual(self.messenger.ping_count, 1)
        data.step()
        self.assertEqual(self.messenger.ping_count, 2)

    # online
    def test_when_online_loop_continues_even_when_no_data(self):
        data = EegData(self.classifier, self.eeg, self.markers, self.messenger)
        data.setup(online=True)
        data.run(max_loops=10)
        self.assertEqual(self.messenger.ping_count, 10)

    def test_online_single_step_runs_single_loop(self):
        data = EegData(self.classifier, self.eeg, self.markers, self.messenger)
        data.setup(online=False)

        data.step()
        self.assertEqual(self.messenger.ping_count, 1)
        data.step()
        self.assertEqual(self.messenger.ping_count, 2)

    def test_when_online_and_invalid_markers_then_loop_continues(self):
        data = EegData(self.classifier, self.eeg, self.markers, self.messenger)

        # provide garbage data (None is invalid, length of data and timestamps doesn't match)
        self.markers.marker_data = [1.0]
        self.markers.marker_timestamps = [1.0]
        data.setup(online=True)
        data.run(max_loops=2)

        # if we didn't crash, sanity check that loops did happen
        self.assertEqual(self.messenger.ping_count, 2)

    def test_when_online_and_invalid_eeg_then_loop_continues(self):
        data = EegData(self.classifier, self.eeg, self.markers, self.messenger)

        # provide garbage data (None is invalid, length of data and timestamps doesn't match)
        self.eeg.eeg_data = [1.0]
        self.eeg.eeg_timestamps = [1.0]
        data.setup(online=True)
        data.run(max_loops=2)

        # if we didn't crash, sanity check that loops did happen
        self.assertEqual(self.messenger.ping_count, 2)


# Placeholder to make EegData happy
class _MockMarkerSource(MarkerSource):
    name = "MockMarker"
    marker_data = [[]]
    marker_timestamps = []

    def get_markers(self) -> tuple[list[list], list]:
        return [self.marker_data, self.marker_timestamps]

    def time_correction(self) -> float:
        return 0.0


# Placeholder to make EegData happy
class _MockEegSource(EegSource):
    name = "MockEeg"
    fsample = 0.0
    n_channels = 0
    channel_types = []
    channel_units = []
    channel_labels = []
    eeg_data = [[]]
    eeg_timestamps = []

    def get_samples(self) -> tuple[list[list], list]:
        return [self.eeg_data, self.eeg_timestamps]

    def time_correction(self) -> float:
        return 0.0


# Count how often these methods are called to check that EegData
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
