import unittest

from pylsl import StreamInfo, StreamOutlet, IRREGULAR_RATE

from bci_essentials.io.lsl_sources import LslMarkerSource, LslEegSource


class TestLslSourceTimeouts(unittest.TestCase):
    def test_lsl_marker_source_raises_exception_on_timeout(self):
        with self.assertRaises(Exception):
            LslMarkerSource(timeout=0)

    def test_lsl_eeg_source_raises_exception_on_timeout(self):
        with self.assertRaises(Exception):
            LslEegSource(timeout=0)


class TestLslMarkerSource(unittest.TestCase):
    def setUp(self) -> None:
        self.sender = LslSender("LSL_Marker_Strings", 1)
        self.source = LslMarkerSource()

    def test_marker_name(self):
        self.assertEqual(self.source.name, "LSL Test")

    # def test_get_marker(self):
    #     self.sender.mark()
    #     samples, timestamps = self.source.get_markers()
    #     self.assertIsNotNone(samples)
    #     self.assertEqual(len(samples), 1)
    #     self.assertEqual(len(timestamps), 1)

    def test_marker_time_correction_is_probably_not_zero(self):
        # this is a bit goofy, but a realy LSL stream probably won't have a correction of zero
        self.assertNotEqual(self.source.time_correction(), 0.0)


class TestLslEegSource(unittest.TestCase):
    def setUp(self) -> None:
        self.sender = LslSender("EEG", 8, 128.0)
        self.source = LslEegSource()

    def test_marker_name(self):
        self.assertEqual(self.source.name, "LSL Test")

    def test_eeg_fsample(self):
        self.assertEqual(self.source.fsample, 128.0)

    def test_eeg_nchannel(self):
        self.assertEqual(self.source.n_channels, 8)

    def test_eeg_channel(self):
        self.assertEqual(len(self.source.channel_types), 8)
        # hard to inject this, going to skip
        # self.assertEqual(self.source.channel_types[0], "eeg")

    def test_eeg_channel_units(self):
        self.assertEqual(len(self.source.channel_units), 8)
        # hard to inject this, going to skip
        # self.assertEqual(self.source.channel_units[0], "microvolts")

    def test_eeg_channel_labels(self):
        self.assertEqual(len(self.source.channel_labels), 8)
        # hard to inject this, going to skip
        # self.assertEqual(self.source.channel_labels[0], "FP1")

    # def test_get_eeg(self):
    #     self.sender.mark()
    #     samples, timestamps = self.source.get_markers()
    #     self.assertIsNotNone(samples)
    #     self.assertEqual(len(samples), 1)
    #     self.assertEqual(len(timestamps), 1)

    def test_eeg_time_correction_is_probably_not_zero(self):
        # this is a bit goofy, but a realy LSL stream probably won't have a correction of zero
        self.assertNotEqual(self.source.time_correction(), 0.0)


class LslSender:
    def __init__(self, streamtype: str, channels: int, rate: float = IRREGULAR_RATE):
        info = StreamInfo(
            "LSL Test", streamtype, channels, rate, channel_format="float32"
        )
        self.__outlet = StreamOutlet(info)
        self.__mark_count: float = 0.0

    # This won't work properly unless running in a separate thread / process
    def mark(self):
        marker = [self.__mark_count]
        self.__outlet.push_sample(marker)
        self.__mark_count += 1.0
