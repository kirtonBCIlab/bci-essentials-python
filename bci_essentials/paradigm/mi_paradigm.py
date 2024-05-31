import numpy as np

from .paradigm import Paradigm


class MiParadigm(Paradigm):
    """
    MI paradigm.
    """

    def __init__(
        self,
        filters=[5, 30],
        iterative_training=False,
        live_update=False,
        buffer_time=0.01,
    ):
        """
        Parameters
        ----------
        filters : list of floats, *optional*
            Filter bands.
            - Default is `[5, 30]`.
        iterative_training : bool, *optional*
            Flag to indicate if the classifier will be updated iteratively.
            - Default is `False`.
        live_update : bool, *optional*
            Flag to indicate if the classifier will be used to provide
            live updates on trial classification.
            - Default is `False`.
        buffer_time : float, *optional*
            Defines the time in seconds after an epoch for which we require EEG data to ensure that all EEG is present in that epoch.
            - Default is `0.01`.
        """
        super().__init__(filters)

        self.live_update = live_update
        self.iterative_training = iterative_training

        if self.live_update:
            self.classify_each_epoch = True
            self.classify_each_trial = False
        else:
            self.classify_each_trial = True
            self.classify_each_epoch = False

        self.buffer_time = buffer_time

    def get_eeg_start_and_end_times(self, markers, timestamps):
        """
        Get the start and end times of the EEG data based on the markers.

        Parameters
        ----------
        markers : list of str
            List of markers.
        timestamps : list of float
            List of timestamps.

        Returns
        -------
        float
            Start time.
        float
            End time.
        """
        start_time = timestamps[0] - self.buffer_time

        end_time = timestamps[-1] + float(markers[-1].split(",")[-1]) + self.buffer_time

        return start_time, end_time

    def process_markers(self, markers, marker_timestamps, eeg, eeg_timestamps, fsample):
        """
        This takes in the markers and EEG data and processes them into epochs accordingt to the MI paradigm.

        Parameters
        ----------
        markers : list of str
            List of markers.
        marker_timestamps : list of float
            List of timestamps.
        eeg : np.array
            EEG data. Shape is (n_channels, n_samples).
        eeg_timestamps : np.array
            EEG timestamps. Shape is (n_samples).
        fsample : float
            Sampling frequency.

        Returns
        -------
        np.array
            Processed EEG data. Shape is (n_epochs, n_channels, n_samples).
        np.array
            Labels. Shape is (n_epochs).
        """

        # Initialize y
        y = np.zeros(len(markers), dtype=int)

        for i, marker in enumerate(markers):
            marker = marker.split(",")
            label = int(marker[2])
            epoch_length = float(marker[3])

            n_channels, _ = eeg.shape

            marker_timestamp = marker_timestamps[i]

            # Subtract the marker timestamp from the EEG timestamps so that 0 becomes the marker onset
            marker_eeg_timestamps = eeg_timestamps - marker_timestamp

            # Create the epoch time vector
            epoch_time = np.arange(0, epoch_length, 1 / fsample)

            # Initialize the EEG data array
            epoch_eeg = np.zeros((1, n_channels, len(epoch_time)))

            # Interpolate the EEG data to the epoch time vector for each channel
            for c in range(n_channels):
                epoch_eeg[0, c, :] = np.interp(
                    epoch_time, marker_eeg_timestamps, eeg[c, :]
                )

            epoch_eeg[0, :, :] = super()._preprocess(
                epoch_eeg[0, :, :], fsample, self.lowcut, self.highcut
            )

            if i == 0:
                X = epoch_eeg
            else:
                X = np.concatenate((X, epoch_eeg), axis=0)

            y[i] = label

        return X, y

    # TODO: Implement this to check compatibility between paradigm and classifier
    def check_compatibility(self):
        pass
