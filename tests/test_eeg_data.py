import unittest

from bci_essentials.io.sources import MarkerSource, EegSource
from bci_essentials.io.messenger import Messenger
from bci_essentials.eeg_data import EEG_data
from bci_essentials.classification.null_classifier import Null_classifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger()


class TestEegData(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = Null_classifier()
        self.eeg = _MockEegSource()
        self.markers = _MockMarkerSource()
        self.messenger = _MockMessenger()

    def test_trg_channel_types_changed_to_stim(self):
        self.eeg.channel_types = ["eeg", "trg", "eeg", "stim"]
        data = EEG_data(self.classifier, self.eeg, self.markers)

        self.assertEqual(data.ch_type, ["eeg", "stim", "eeg", "stim"])

    def test_dsi7_mods(self):
        self.eeg.name = "DSI7"
        self.eeg.channel_labels = ["foo"] * 8
        self.eeg.nchannels = 8
        data = EEG_data(self.classifier, self.eeg, self.markers)

        self.assertEqual(data.nchannels, 7)
        self.assertEqual(len(data.channel_labels), 7)

    def test_dsi24_mods(self):
        self.eeg.name = "DSI24"
        self.eeg.channel_labels = ["foo"] * 24
        self.eeg.nchannels = 24
        data = EEG_data(self.classifier, self.eeg, self.markers)

        self.assertEqual(data.nchannels, 23)
        self.assertEqual(len(data.channel_labels), 23)

    def test_ping_sent_with_each_loop(self):
        data = EEG_data(self.classifier, self.eeg, self.markers, self.messenger)
        data.main(online=True, max_loops=10)

        self.assertEqual(self.messenger.ping_count, 10)

    # def test_when_online_and_invalid_markers_then_loop_continues(self):
    #     data = EEG_data(self.classifier, self.eeg, self.markers, self.messenger)

    #     self.markers.marker_data = None
    #     self.markers.marker_timestamps = []
    #     data.main(online=True, max_loops=2)
    #     self.assertEqual(self.messenger.ping_count, 2)

    #     self.markers.marker_data = []
    #     self.markers.marker_timestamps = []
    #     data.main(online=True, max_loops=2)
    #     self.assertEqual(self.messenger.ping_count, 4)

    # def test_when_online_and_invalid_eeg_then_loop_continues(self):
    #     data = EEG_data(self.classifier, self.eeg, self.markers, self.messenger)

    #     self.eeg.eeg_data = None
    #     self.eeg.eeg_timestamps = []
    #     data.main(online=True, max_loops=2)
    #     self.assertEqual(self.messenger.ping_count, 2)

    #     self.eeg.eeg_data = []
    #     self.eeg.eeg_timestamps = []
    #     data.main(online=True, max_loops=2)
    #     self.assertEqual(self.messenger.ping_count, 4)


# Placeholder to make EEG_data happy
class _MockMarkerSource(MarkerSource):
    name = "MockMarker"
    marker_data = [[], []]
    marker_timestamps = []

    def get_markers(self) -> tuple[list, list]:
        return [self.marker_data, self.marker_timestamps]

    def time_correction(self) -> float:
        return 0.0


# Placeholder to make EEG_data happy
class _MockEegSource(EegSource):
    name = "MockEeg"
    fsample = 0.0
    nchannels = 0
    channel_types = []
    channel_units = []
    channel_labels = []
    eeg_data = [[], []]
    eeg_timestamps = []

    def get_samples(self) -> tuple[list, list]:
        return [self.eeg_data, self.eeg_timestamps]

    def time_correction(self) -> float:
        return 0.0


# Count how often these methods are called to check that EEG_data
# would have sent a message, not checking content.
class _MockMessenger(Messenger):
    ping_count = 0
    marker_received_count = 0
    prediction_count = 0

    def ping(self):
        self.ping_count += 1

    def marker_received(self, marker):
        self.marker_received_count += 1

    def prediction(self, prediction):
        self.prediction_count += 1
