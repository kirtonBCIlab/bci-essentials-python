import numpy as np

from .base_paradigm import BaseParadigm

class MiParadigm(BaseParadigm):
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
        start_time = timestamps[0] - self.buffer_time

        end_time = timestamps[-1] + float(markers[-1].split(",")[-1]) + self.buffer_time

        return start_time, end_time

    def process_markers(self, markers, marker_timestamps, eeg, eeg_timestamps, fsample):
        """
        This takes in the markers and EEG data and processes them into epochs.
        """

        for i, marker in enumerate(markers):
            marker = marker.split(",")
            paradigm_string = marker[0]  # Maybe use this as a compatibility check?
            num_options = int(marker[1])
            label = marker[2]
            epoch_length = float(marker[3])

            nchannels, _ = eeg.shape

            marker_timestamp = marker_timestamps[i]

            # Subtract the marker timestamp from the EEG timestamps so that 0 becomes the marker onset
            marker_eeg_timestamps = eeg_timestamps - marker_timestamp

            # Create the epoch time vector
            epoch_time = np.arange(0, epoch_length, 1 / fsample)

            X = np.zeros((1, nchannels, len(epoch_time)))
            y = None

            # Interpolate the EEG data to the epoch time vector for each channel
            for c in range(nchannels):
                X[i, c, :] = np.interp(epoch_time, marker_eeg_timestamps, eeg[c, :])

            X[i, :, :] = super()._preprocess(
                X[i, :, :], fsample, self.lowcut, self.highcut
            )

            if i == 0:
                X = X
                y = np.array([int(label)])
            else:
                X = np.concatenate((X, X), axis=0)
                y = np.concatenate((y, int(label)))

        return X, y

    # TODO: Implement this
    def check_compatibility(self):
        pass
