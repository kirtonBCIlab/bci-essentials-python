import numpy as np
from ..signal_processing import bandpass


# Will eventually move somewhere else
class DataTank:
    """
    Class shaping EEG data into trials for classification.

    Please use a subclass, depending on your paradigm.
    """

    def __init__(self):
        self.raw_eeg = np.zeros((0, 0))
        self.raw_eeg_timestamps = np.zeros((0))
        self.raw_marker_strings = np.zeros((0), dtype=str)
        self.raw_marker_timestamps = np.zeros((0))
        self.event_marker_strings = np.zeros((0), dtype=str)
        self.event_marker_timestamps = np.zeros((0))

        self.live_classification = False

        self.latest_eeg_timestamp = 0

        self.epochs_sent = 0
        self.epochs = np.zeros((0, 0))

    def set_source_data(
        self, headset_string, fsample, n_channels, ch_types, ch_units, channel_labels
    ):
        """ """
        self.headset_string = headset_string
        self.fsample = fsample
        self.n_channels = n_channels
        self.ch_types = ch_types
        self.ch_units = ch_units
        self.channel_labels = channel_labels

    def package_resting_state_data(self):
        pass

    def add_raw_eeg(self, new_raw_eeg, new_eeg_timestamps):
        # If this is the first chunk of EEG, initialize the arrays
        if self.raw_eeg.size == 0:
            self.raw_eeg = new_raw_eeg
            self.raw_eeg_timestamps = new_eeg_timestamps
        else:
            self.raw_eeg = np.concatenate((self.raw_eeg, new_raw_eeg))
            self.raw_eeg_timestamps = np.concatenate(
                (self.raw_eeg_timestamps, new_eeg_timestamps)
            )

        self.latest_eeg_timestamp = new_eeg_timestamps[-1]

    def add_raw_markers(self, new_marker_strings, new_marker_timestamps):
        if self.raw_marker_strings.size == 0:
            self.raw_marker_strings = new_marker_strings
            self.raw_marker_timestamps = new_marker_timestamps
        else:
            self.raw_marker_strings = np.concatenate(
                (self.raw_marker_strings, new_marker_strings)
            )
            self.raw_marker_timestamps = np.concatenate(
                (self.raw_marker_timestamps, new_marker_timestamps)
            )

    def get_raw_eeg(self):
        # Get the EEG data between the start and end times
        return self.raw_eeg, self.raw_eeg_timestamps

    def get_raw_markers(self):
        return self.raw_marker_strings, self.raw_marker_timestamps

    def add_epochs(self, X, y):
        # Add new epochs to the data tank
        if self.epochs.size == 0:
            self.epochs = np.array(X)
            self.labels = np.array(y)
        else:
            self.epochs = np.concatenate((self.epochs, np.array(X)))
            self.labels = np.concatenate((self.labels, np.array(y)))

    def get_epochs(self, latest=False):
        if latest:
            # Return only the new data
            first_unsent = self.epochs_sent
            self.epochs_sent = len(self.epochs)

            return self.epochs[first_unsent:], self.labels[first_unsent:]
        else:
            # Return all
            return self.epochs, self.labels

    def save_raw():
        pass

    def save_epochs_as_npz(self, file_name: str):
        """
        TODO - replace this with npz saving of epochs in data tank
        Saves EEG trials and labels as a numpy file.

        Parameters
        ----------
        file_name : str
            The name of the file to save the EEG trials and labels to.

        Returns
        -------
        `None`

        """
        # # Check if file ends with .npz, if not add it
        # if file_name[-4:] != ".npz":
        #     file_name += ".npz"

        # # Get the raw EEG trials and labels
        # X = self.raw_eeg_trials
        # y = self.labels

        # # Cut X and y to be the lenght of the number of trials, because X and y are initialized to be the maximum number of trials
        # X = X[: self.n_trials]
        # y = y[: self.n_trials]

        # # Save the raw EEG trials and labels as a numpy file
        # np.savez(file_name, X=X, y=y)
