import unittest

from bci_essentials.sources.sources import MarkerSource, EegSource
from bci_essentials.eeg_data import EEG_data
from bci_essentials.classification.null_classifier import Null_classifier


class TestEegData(unittest.TestCase):
    def test_trg_channel_types_changed_to_stim(self):
        classifier = Null_classifier()
        eeg = MockEegSource()
        markers = MockMarkerSource()

        eeg.channel_types = ["eeg", "trg", "eeg", "stim"]
        data = EEG_data(classifier, eeg, markers)
        self.assertEqual(data.ch_type, ["eeg", "stim", "eeg", "stim"])

    def test_dsi7_mods(self):
        classifier = Null_classifier()
        eeg = MockEegSource()
        markers = MockMarkerSource()

        eeg.name = "DSI7"
        eeg.channel_labels = ["foo"] * 8
        eeg.nchannels = 8
        data = EEG_data(classifier, eeg, markers)
        self.assertEqual(data.nchannels, 7)
        self.assertEqual(len(data.channel_labels), 7)

    def test_dsi24_mods(self):
        classifier = Null_classifier()
        eeg = MockEegSource()
        markers = MockMarkerSource()

        eeg.name = "DSI24"
        eeg.channel_labels = ["foo"] * 24
        eeg.nchannels = 24
        data = EEG_data(classifier, eeg, markers)
        self.assertEqual(data.nchannels, 23)
        self.assertEqual(len(data.channel_labels), 23)


class MockMarkerSource(MarkerSource):
    name = "MockMarker"

    def get_markers(self) -> tuple[list, list]:
        return [[], []]

    def time_correction(self) -> float:
        return 0.0


class MockEegSource(EegSource):
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
