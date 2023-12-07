import unittest
import os
from bci_essentials.sources.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger()


class TestXdfSourceNonexistentFiles(unittest.TestCase):
    def test_xdf_marker_source_raises_exception_on_bad_filename(self):
        with self.assertRaises(Exception):
            logger.critical("XDF Marker Source: File doesn't exist")
            XdfMarkerSource("./doesnt_exist.xdf")

    def test_xdf_eeg_source_raises_exception_on_bad_filename(self):
        with self.assertRaises(Exception):
            logger.critical("XDF EEG Source: File doesn't exist")
            XdfEegSource("./doesnt_exist.xdf")


class TestXdfMarkerSource(unittest.TestCase):
    def setUp(self) -> None:
        filepath = os.path.join("examples", "data", "rs_example.xdf")
        self.source = XdfMarkerSource(filepath)

    def test_marker_name(self):
        self.assertEqual(self.source.name, "UnityMarkerStream")

    def test_marker_get_samples_provides_all_samples_in_one_go(self):
        # first get is all the samples
        samples, timestamps = self.source.get_markers()
        self.assertIsNotNone(samples)
        self.assertGreater(len(samples), 0)
        self.assertGreater(len(timestamps), 0)

        # second get is empty
        samples, timestamps = self.source.get_markers()
        self.assertEqual(samples, [])
        self.assertEqual(len(timestamps), 0)

    def test_marker_time_correction_is_zero_for_xdf(self):
        self.assertEqual(self.source.time_correction(), 0.0)


class TestXdfEegSource(unittest.TestCase):
    def setUp(self) -> None:
        filepath = os.path.join("examples", "data", "rs_example.xdf")
        self.source = XdfEegSource(filepath)

    def test_eeg_name(self):
        self.assertEqual(self.source.name, "g.USBamp-1")

    def test_eeg_fsample(self):
        self.assertEqual(self.source.fsample, 256.0)

    def test_eeg_nchannel(self):
        self.assertEqual(self.source.nchannels, 16)

    def test_eeg_channel(self):
        self.assertEqual(len(self.source.channel_types), 16)
        self.assertEqual(self.source.channel_types[0], "eeg")

    def test_eeg_channel_units(self):
        self.assertEqual(len(self.source.channel_units), 16)
        self.assertEqual(self.source.channel_units[0], "microvolts")

    def test_eeg_channel_labels(self):
        self.assertEqual(len(self.source.channel_labels), 16)
        self.assertEqual(self.source.channel_labels[0], "FP1")

    def test_eeg_get_samples_provides_all_samples_in_one_go(self):
        # first get is all the samples
        samples, timestamps = self.source.get_samples()
        self.assertIsNotNone(samples)
        self.assertGreater(len(samples), 0)
        self.assertGreater(len(timestamps), 0)
        self.assertGreater(samples.shape[0], 0)

        # second get is empty
        samples, timestamps = self.source.get_samples()
        self.assertEqual(samples, [])
        self.assertEqual(len(timestamps), 0)

    def test_eeg_time_correction_is_zero_for_xdf(self):
        self.assertEqual(self.source.time_correction(), 0.0)


if __name__ == "__main__":
    unittest.main()
