import numpy as np


# Will eventually move somewhere else
class DataTank:
    """
    DataTank class is for storing raw EEG, markers, epochs, and rtesting state data

    TODO: Add your desired flavour of save output from here.
    To be added:
    - MNE Raw
    - MNE Epochs
    - BIDS
    - XDF
    """

    def __init__(self):
        """
        Initialize the DataTank.
        """

        # Initialize np arrays to store the data
        self.__raw_eeg = np.zeros((0, 0))
        self.__raw_eeg_timestamps = np.zeros((0))
        self.__raw_marker_strings = np.zeros((0), dtype=str)
        self.__raw_marker_timestamps = np.zeros((0))
        # self.event_marker_strings = np.zeros((0), dtype=str)
        # self.event_marker_timestamps = np.zeros((0))

        # Keep track of the latest timestamp
        self.latest_eeg_timestamp = 0

        # Keep track of how many epochs have been sent, so it is possible to only send new ones
        self.epochs_sent = 0
        self.epochs = np.zeros((0, 0))

    def set_source_data(
        self, headset_string, fsample, n_channels, ch_types, ch_units, channel_labels
    ):
        """
        Set the source data for the DataTank so that this metadata can be saved with the data.

        Parameters
        ----------

        headset_string : str
            The name of the headset used to collect the data.
        fsample : float
            The sampling frequency of the data.
        n_channels : int
            The number of channels in the data.
        ch_types : list of str
            The type of each channel.
        ch_units : list of str
            The units of each channel.
        channel_labels : list of str
            The labels of each channel.

        Returns
        -------
        `None`
        """
        self.headset_string = headset_string
        self.fsample = fsample
        self.n_channels = n_channels
        self.ch_types = ch_types
        self.ch_units = ch_units
        self.channel_labels = channel_labels

    def add_raw_eeg(self, new_raw_eeg, new_eeg_timestamps):
        """
        Add raw EEG data to the data tank.

        Parameters
        ----------
        new_raw_eeg : np.array
            The new raw EEG data to add.

        new_eeg_timestamps : np.array
            The timestamps of the new raw EEG data.

        Returns
        -------
        `None`
        """

        # If this is the first chunk of EEG, initialize the arrays
        if self.__raw_eeg.size == 0:
            self.__raw_eeg = new_raw_eeg
            self.__raw_eeg_timestamps = new_eeg_timestamps
        else:
            self.__raw_eeg = np.concatenate((self.__raw_eeg, new_raw_eeg), axis=1)
            self.__raw_eeg_timestamps = np.concatenate(
                (self.__raw_eeg_timestamps, new_eeg_timestamps)
            )

        self.latest_eeg_timestamp = new_eeg_timestamps[-1]

    def add_raw_markers(self, new_marker_strings, new_marker_timestamps):
        """
        Add raw markers to the data tank.

        Parameters
        ----------
        new_marker_strings : np.array
            The new marker strings to add.
        new_marker_timestamps : np.array
            The timestamps of the new marker strings.

        Returns
        -------
        `None`
        """

        if self.__raw_marker_strings.size == 0:
            self.__raw_marker_strings = new_marker_strings
            self.__raw_marker_timestamps = new_marker_timestamps
        else:
            self.__raw_marker_strings = np.concatenate(
                (self.__raw_marker_strings, new_marker_strings)
            )
            self.__raw_marker_timestamps = np.concatenate(
                (self.__raw_marker_timestamps, new_marker_timestamps)
            )

    def get_raw_eeg(self):
        """
        Get the raw EEG data from the DataTank.

        Returns
        -------
        np.array
            The raw EEG data.
        np.array
            The timestamps of the raw EEG data.
        """
        # Get the EEG data between the start and end times
        return self.__raw_eeg, self.__raw_eeg_timestamps

    def get_raw_markers(self):
        """
        Get the raw markers from the DataTank.

        Returns
        -------
        np.array
            The raw marker strings.
        np.array
            The timestamps of the raw marker strings.
        """
        return self.__raw_marker_strings, self.__raw_marker_timestamps

    def add_epochs(self, X, y):
        """
        Add epochs to the data tank.

        Parameters
        ----------
        X : np.array
            The new epochs to add. Shape is (n_epochs, n_channels, n_samples).
        y : np.array
            The labels of the epochs. Shape is (n_epochs).

        Returns
        -------
        `None`

        """
        # Add new epochs to the data tank
        if self.epochs.size == 0:
            self.epochs = np.array(X)
            self.labels = np.array(y)
        else:
            # Check the size of the new data
            if X.shape[1:] != self.epochs.shape[1:]:
                print(
                    "Epochs are not the same size, skipping this data.",
                )
            else:
                self.epochs = np.concatenate((self.epochs, np.array(X)))
                self.labels = np.concatenate((self.labels, np.array(y)))

    def get_epochs(self, latest=False):
        """
        Get the epochs from the data tank.

        Parameters
        ----------
        latest : bool
            If `True`, only return the new data since the last call to this function.

        Returns
        -------
        np.array
            The epochs. Shape is (n_epochs, n_channels, n_samples).
        np.array
            The labels of the epochs. Shape is (n_epochs).

        """
        if latest:
            # Return only the new data
            first_unsent = self.epochs_sent
            self.epochs_sent = len(self.epochs)

            return self.epochs[first_unsent:], self.labels[first_unsent:]
        else:
            # Return all
            return self.epochs, self.labels

    def add_resting_state_data(self, resting_state_data):
        """
        Add resting state data to the data tank.

        Parameters
        ----------
        resting_state_data : dict
            Dictionary containing resting state data.

        Returns
        -------
        `None`

        """
        # Get the resting state data
        self.__resting_state_data = resting_state_data

    def get_resting_state_data(self):
        """
        Get the resting state data.

        Returns
        -------
        dict
            Dictionary containing resting state data.

        """
        return self.__resting_state_data

    def save_epochs_as_npz(self, file_name: str):
        """
        Saves EEG trials and labels as a numpy file.

        Parameters
        ----------
        file_name : str
            The name of the file to save the EEG trials and labels to.

        Returns
        -------
        `None`

        """
        # Check if file ends with .npz, if not add it
        if file_name[-4:] != ".npz":
            file_name += ".npz"

        # Get the raw EEG trials and labels
        X = self.epochs
        y = self.labels

        # Save the raw EEG trials and labels as a numpy file
        np.savez(file_name, X=X, y=y)
