import unittest

from bci_essentials.io.sources import MarkerSource, EegSource
from bci_essentials.io.messenger import Messenger
from bci_essentials.eeg_data import EegData
from bci_essentials.classification.null_classifier import Null_classifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger()


class TestEegData(unittest.TestCase):
    def test_trg_channel_types_changed_to_stim(self):
        classifier = Null_classifier()
        eeg = _MockEegSource()
        markers = _MockMarkerSource()

        eeg.channel_types = ["eeg", "trg", "eeg", "stim"]
        data = EegData(classifier, eeg, markers)

        self.assertEqual(data.ch_type, ["eeg", "stim", "eeg", "stim"])

    def test_dsi7_mods(self):
        classifier = Null_classifier()
        eeg = _MockEegSource()
        markers = _MockMarkerSource()

        eeg.name = "DSI7"
        eeg.channel_labels = ["foo"] * 8
        eeg.nchannels = 8
        data = EegData(classifier, eeg, markers)

        self.assertEqual(data.nchannels, 7)
        self.assertEqual(len(data.channel_labels), 7)

    def test_dsi24_mods(self):
        classifier = Null_classifier()
        eeg = _MockEegSource()
        markers = _MockMarkerSource()

        eeg.name = "DSI24"
        eeg.channel_labels = ["foo"] * 24
        eeg.nchannels = 24
        data = EegData(classifier, eeg, markers)

        self.assertEqual(data.nchannels, 23)
        self.assertEqual(len(data.channel_labels), 23)

    def test_ping_sent_with_each_loop(self):
        classifier = Null_classifier()
        eeg = _MockEegSource()
        markers = _MockMarkerSource()
        messenger = _MockMessenger()

        data = EegData(classifier, eeg, markers, messenger)
        data.main(online=True, max_loops=3)

        self.assertEqual(messenger.ping_count, 3)


# Placeholder to make EegData happy
class _MockMarkerSource(MarkerSource):
    name = "MockMarker"

    def get_markers(self) -> tuple[list, list]:
        return [[], []]

    def time_correction(self) -> float:
        return 0.0


# Placeholder to make EegData happy
class _MockEegSource(EegSource):
    name = "MockEeg"
    fsample = 0.0
    nchannels = 0
    channel_types = []
    channel_units = []
    channel_labels = []

    def get_samples(self) -> tuple[list, list]:
        return [[], []]

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

    def prediction(self, prediction):
        self.prediction_count += 1
