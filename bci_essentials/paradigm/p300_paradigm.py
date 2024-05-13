import numpy as np

from .base_paradigm import BaseParadigm

class P300Paradigm(BaseParadigm):
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
            flash_type = marker[1] # s=single, m=multiple
            num_options = int(marker[2])
            train_target = int(marker[3])
            flash_indices = [int(x) for x in marker[4:]]

            nchannels, _ = eeg.shape
            epoch_length = self.epoch_end - self.epoch_start
            nsamples = np.ceil(epoch_length * fsample)

            marker_timestamp = marker_timestamps[i]

            # Subtract the marker timestamp from the EEG timestamps so that 0 becomes the marker onset
            eeg_timestamps = eeg_timestamps - marker_timestamp

            # Create the epoch time vector
            epoch_time = np.arange(0, epoch_length, 1 / fsample)

            X = np.zeros((1, nchannels, len(epoch_time)))
            y = None

            # Interpolate the EEG data to the epoch time vector for each channel
            for c in range(nchannels):
                X[i, c, :] = np.interp(epoch_time, eeg_timestamps, eeg[c, :])

            X[i, :, :] = super()._preprocess(
                X[i, :, :], fsample, self.lowcut, self.highcut
            )

            if i == 0:
                X = X
                y = label
            else:
                X = np.concatenate((X, X), axis=0)
                y = np.concatenate((y, label))

        return X, y

    # TODO: Implement this
    def check_compatibility(self):
        pass