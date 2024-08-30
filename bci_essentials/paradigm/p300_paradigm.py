import numpy as np

from .paradigm import Paradigm


class P300Paradigm(Paradigm):
    """
    P300 paradigm.
    """

    def __init__(
        self,
        filters=[1, 15],
        iterative_training=False,
        epoch_start=0,
        epoch_end=0.6,
        buffer_time=0.01,
    ):
        """
        Parameters
        ----------
        filters : list of floats, *optional*
            Filter bands.
            - Default is `[1, 15]`.
        iterative_training : bool, *optional*
            Flag to indicate if the classifier will be updated iteratively.
            - Default is `False`.
        epoch_start : float, *optional*
            The start of the epoch relative to flash onset in seconds.
            - Default is `0`.
        epoch_end : float, *optional*
            The end of the epoch relative to flash onset in seconds.
            - Default is `0.6`.
        buffer_time : float, *optional*
            Defines the time in seconds after an epoch for which we require EEG data to ensure that all EEG is present in that epoch.
            - Default is `0.01`.
        """

        super().__init__(filters)

        self.iterative_training = iterative_training

        # The P300 paradigm needs epochs from each object in order to decide which object was selected
        # therefore we need to classify each trial
        self.classify_each_trial = True
        self.classify_each_epoch = False

        # This paradigm uses ensemble averaging to increase the signal to nosie ratio and will average
        # over all epochs for each object by default
        self.ensemble_average = True

        self.epoch_start = epoch_start
        self.epoch_end = epoch_end

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
        start_time = timestamps[0] + self.epoch_start - self.buffer_time

        end_time = timestamps[-1] + self.epoch_end + self.buffer_time

        return start_time, end_time

    def process_markers(self, markers, marker_timestamps, eeg, eeg_timestamps, fsample):
        """
        This takes in the markers and EEG data and processes them into epochs according to the P300 paradigm.

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

        n_channels, _ = eeg.shape
        num_objects = int(markers[0].split(",")[2])

        train_target = int(markers[3].split(",")[3])
        y = np.zeros(num_objects, dtype=int)
        if train_target is not -1:
            y[train_target] = 1
        if train_target is -1:  # Set all values of y to -1
            y = np.full(num_objects, -1)

        flash_counts = np.zeros(num_objects)

        # X = np.zeros((num_objects, n_channels, len(epoch_time)))

        # Do ensemble averaging so that we return a single epoch for each object

        for i, marker in enumerate(markers):
            marker = marker.split(",")
            flash_indices = [int(x) for x in marker[4:]]

            n_channels, _ = eeg.shape

            marker_timestamp = marker_timestamps[i]

            # Subtract the marker timestamp from the EEG timestamps so that 0 becomes the marker onset
            marker_eeg_timestamps = eeg_timestamps - marker_timestamp

            # Create the epoch time vector
            epoch_time = np.arange(self.epoch_start, self.epoch_end, 1 / fsample)

            epoch_X = np.zeros((1, n_channels, len(epoch_time)))

            # Initialize object_epochs if this is the first epoch
            if i == 0:
                object_epochs = [
                    np.zeros((num_objects, n_channels, len(epoch_time)))
                ] * num_objects

            # Interpolate the EEG data to the epoch time vector for each channel
            for c in range(n_channels):
                epoch_X[0, c, :] = np.interp(
                    epoch_time, marker_eeg_timestamps, eeg[c, :]
                )

            epoch_X[0, :, :] = super()._preprocess(
                epoch_X[0, :, :], fsample, self.lowcut, self.highcut
            )

            # For each flash index in the marker
            for flash_index in flash_indices:
                if flash_counts[flash_index] == 0:
                    object_epochs[flash_index] = epoch_X
                    flash_counts[flash_index] += 1
                else:
                    object_epochs[flash_index] = np.concatenate(
                        (object_epochs[flash_index], epoch_X), axis=0
                    )
                    flash_counts[flash_index] += 1

        # Average all epochs for each object
        object_epochs_mean = [np.zeros((n_channels, len(epoch_time)))] * num_objects
        for i in range(num_objects):
            object_epochs_mean[i] = np.mean(object_epochs[i], axis=0)

        X = np.zeros((num_objects, n_channels, len(epoch_time)))
        for i in range(num_objects):
            X[i, :, :] = object_epochs_mean[i]
        # # object_epochs_mean = np.mean(object_epochs, axis=1)
        # X = object_epochs_mean

        return X, y

    # TODO: Implement this
    def check_compatibility(self):
        pass
