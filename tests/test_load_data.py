import unittest
from bci_essentials.bci_data import EEG_data


class TestLoadData(unittest.TestCase):
    def test_load_rs_xdf(self):
        # Get the rs example data from /examples/data
        rs_xdf_path = "examples//data//rs_example.xdf"

        # Load the data
        rs_data = EEG_data()
        rs_data.load_offline_eeg_data(filename=rs_xdf_path, print_output=False)

        # Check that fsample > 0
        self.assertGreater(rs_data.fsample, 0)

        # Check that the dimensions of the EEG data are non-zero
        self.assertGreater(rs_data.eeg_data.shape[0], 0)

        # Check that the EEG data dimensions are (n_timestamps, n_channels))
        n_timestamps = len(rs_data.eeg_timestamps)
        n_channels = len(rs_data.channel_labels)

        self.assertEqual(rs_data.eeg_data.shape, (n_timestamps, n_channels))


if __name__ == "__main__":
    unittest.main()
