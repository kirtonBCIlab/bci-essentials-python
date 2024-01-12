import unittest
import os.path
from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource


class TestXdfSourceNonexistentFiles(unittest.TestCase):
    def test_xdf_marker_source_raises_exception_on_bad_filename(self):
        with self.assertRaises(Exception):
            XdfMarkerSource("doesnt_exist.xdf")

    def test_xdf_eeg_source_raises_exception_on_bad_filename(self):
        with self.assertRaises(Exception):
            XdfEegSource("doesnt_exist.xdf")


class TestXdfMarkerSource(unittest.TestCase):
    def setUp(self) -> None:
        filepath = os.path.join(os.path.dirname(__file__), "data", "xdf_test_data.xdf")
        self.source = XdfMarkerSource(filepath)

    def test_marker_name(self):
        self.assertEqual(self.source.name, "UnityMarkerStream")

    def test_marker_get_samples_provides_all_samples_in_one_go(self):
        # first get is all the samples, ensure shape of array is: [n samples, 1]
        samples, timestamps = self.source.get_markers()
        self.assertIsNotNone(samples)
        self.assertEqual(len(samples), 36)
        self.assertEqual(len(timestamps), 36)
        self.assertEqual(len(samples[0]), 1)

        # second get is empty
        samples, timestamps = self.source.get_markers()
        self.assertEqual(samples, [[]])
        self.assertEqual(len(timestamps), 0)

    def test_marker_time_correction_is_zero_for_xdf(self):
        self.assertEqual(self.source.time_correction(), 0.0)


class TestXdfEegSource(unittest.TestCase):
    def setUp(self) -> None:
        filepath = os.path.join(os.path.dirname(__file__), "data", "xdf_test_data.xdf")
        self.source = XdfEegSource(filepath)

    def test_eeg_name(self):
        self.assertEqual(self.source.name, "g.USBamp-1")

    def test_eeg_fsample(self):
        self.assertEqual(self.source.fsample, 256.0)

    def test_eeg_nchannel(self):
        self.assertEqual(self.source.n_channels, 16)

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
        # first get is all the samples, ensure shape of array is: [n samples, n_channels]
        samples, timestamps = self.source.get_samples()
        self.assertIsNotNone(samples)
        self.assertEqual(len(samples), 116992)
        self.assertEqual(len(timestamps), 116992)
        self.assertEqual(len(samples[0]), 16)

        # second get is empty
        samples, timestamps = self.source.get_samples()
        self.assertEqual(samples, [[]])
        self.assertEqual(len(timestamps), 0)

    def test_eeg_time_correction_is_zero_for_xdf(self):
        self.assertEqual(self.source.time_correction(), 0.0)


if __name__ == "__main__":
    unittest.main()
