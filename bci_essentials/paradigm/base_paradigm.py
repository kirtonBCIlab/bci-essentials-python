import numpy as np

from ..utils.logger import Logger
from ..signal_processing import bandpass

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)

class BaseParadigm():
    def __init__(self, filters=[5,30], channel_subset=None):
        """
        Base class for all paradigms.
        
        Parameters
        ----------
        filters : list of floats | [5, 30]
            Filter bands.
        channel_subset : list of str | None 
            Channel subset to use. 
            """
        self.lowcut = filters[0]
        self.highcut = filters[1]
        self.channel_subset = channel_subset

    def __preprocess(self, eeg, fsample, lowcut, highcut):
        """
        Preprocess EEG data with bandpass filter.
        """
        new_eeg = bandpass(eeg, lowcut, highcut, 5, fsample)

        return new_eeg

    def __interpolate(self, eeg, timestamps, fsample):
        """
        Interpolate EEG data to a uniform sampling rate.
        
        Parameters
        ----------
        eeg : ndarray
            EEG data (nchannels x nsamples).
        timestamps : ndarray
            Timestamps of the EEG data.
        fsample : float
            Sampling rate.
        """
        # Get the number of channels and samples
        n_channels, n_samples = eeg.shape

        # Get the new timestamps
        adjusted_timestamps = timestamps - timestamps[0]
        new_timestamps = np.arange(0, timestamps[-1], 1 / fsample)
        new_eeg = np.zeros((n_channels, len(new_timestamps)))

        # Interpolate the EEG data
        for c in range(n_channels):
            new_eeg[c, :] = np.interp(new_timestamps, adjusted_timestamps, eeg[c, :])

        return new_eeg, new_timestamps
        
    def _package_resting_state_data(self, marker_data, marker_timestamps, eeg_data, eeg_timestamps):
        """Package resting state data.

        Returns
        -------
        `None`
            `self.rest_trials` is updated.

        """
        try:
            logger.debug("Packaging resting state data")

            eyes_open_start_time = []
            eyes_open_end_time = []
            eyes_closed_start_time = []
            eyes_closed_end_time = []
            rest_start_time = []
            rest_end_time = []

            # Initialize start and end locations
            eyes_open_start_loc = []
            eyes_open_end_loc = []
            eyes_closed_start_loc = []
            eyes_closed_end_loc = []
            rest_start_loc = []
            rest_end_loc = []

            current_time = eeg_timestamps[0]
            current_timestamp_loc = 0

            for i in range(len(marker_data)):
                # Get current resting state data marker and time stamp
                current_rs_data_marker = marker_data[i][0]
                current_rs_timestamp = marker_timestamps[i]

                # Increment the EEG until just past the marker timestamp
                while current_time < current_rs_timestamp:
                    current_timestamp_loc += 1
                    current_time = eeg_timestamps[current_timestamp_loc]

                # get eyes open start times
                if current_rs_data_marker == "Start Eyes Open RS: 1":
                    eyes_open_start_time.append(current_rs_timestamp)
                    eyes_open_start_loc.append(current_timestamp_loc - 1)
                    logger.debug("received eyes open start")

                # get eyes open end times
                if current_rs_data_marker == "End Eyes Open RS: 1":
                    eyes_open_end_time.append(current_rs_timestamp)
                    eyes_open_end_loc.append(current_timestamp_loc)
                    logger.debug("received eyes open end")

                # get eyes closed start times
                if current_rs_data_marker == "Start Eyes Closed RS: 2":
                    eyes_closed_start_time.append(current_rs_timestamp)
                    eyes_closed_start_loc.append(current_timestamp_loc - 1)
                    logger.debug("received eyes closed start")

                # get eyes closed end times
                if current_rs_data_marker == "End Eyes Closed RS: 2":
                    eyes_closed_end_time.append(current_rs_timestamp)
                    eyes_closed_end_loc.append(current_timestamp_loc)
                    logger.debug("received eyes closed end")

                # get rest start times
                if current_rs_data_marker == "Start Rest for RS: 0":
                    rest_start_time.append(current_rs_timestamp)
                    rest_start_loc.append(current_timestamp_loc - 1)
                    logger.debug("received rest start")
                # get rest end times
                if current_rs_data_marker == "End Rest for RS: 0":
                    rest_end_time.append(current_rs_timestamp)
                    rest_end_loc.append(current_timestamp_loc)
                    logger.debug("received rest end")

            # Eyes open
            # Get duration, nsamples

            if len(eyes_open_end_loc) > 0:
                duration = np.floor(eyes_open_end_time[0] - eyes_open_start_time[0])
                n_samples = int(duration * self.fsample)

                self.eyes_open_timestamps = np.array(range(n_samples)) / self.fsample
                self.eyes_open_trials = np.ndarray(
                    (len(eyes_open_start_time), self.n_channels, n_samples)
                )
                # Now copy EEG for these trials
                for i in range(len(eyes_open_start_time)):
                    # Get current eyes open start and end locations
                    current_eyes_open_start = eyes_open_start_loc[i]
                    current_eyes_open_end = eyes_open_end_loc[i]

                    new_eeg, new_timestamps = self.__interpolate(eeg_data[:, current_eyes_open_start:current_eyes_open_end], self.eyes_open_timestamps, self.fsample)

                    self.eyes_open_trials[i, :, :] = new_eeg
                    self.eyes_open_timestamps = new_timestamps

            logger.debug("Done packaging resting state data")

            # Eyes closed

            if len(eyes_closed_end_loc) > 0:
                # Get duration, nsmaples
                duration = np.floor(eyes_closed_end_time[0] - eyes_closed_start_time[0])
                n_samples = int(duration * self.fsample)

                eyes_closed_timestamps = np.array(range(n_samples)) / self.fsample
                eyes_closed_trials = np.ndarray(
                    (len(eyes_closed_start_time), self.n_channels, n_samples)
                )
                # Now copy EEG for these trials
                for i in range(len(eyes_closed_start_time)):
                    # Get current eyes closed start and end locations
                    current_eyes_closed_start = eyes_closed_start_loc[i]
                    current_eyes_closed_end = eyes_closed_end_loc[i]

                    new_eeg, new_timestamps = self.__interpolate(eeg_data[:, current_eyes_closed_start:current_eyes_closed_end], eyes_closed_timestamps, self.fsample)

                    eyes_closed_trials[i, :, :] = new_eeg
                    eyes_closed_timestamps = new_timestamps

            # Rest
            if len(rest_end_loc) > 0:
                # Get duration, nsmaples
                while rest_end_time[0] < rest_start_time[0]:
                    rest_end_time.pop(0)
                    rest_end_loc.pop(0)

                duration = np.floor(rest_end_time[0] - rest_start_time[0])

                n_samples = int(duration * self.fsample)

                rest_timestamps = np.array(range(n_samples)) / self.fsample
                rest_trials = np.ndarray(
                    (len(rest_start_time), self.n_channels, n_samples)
                )
                # Now copy EEG for these trials
                for i in range(len(rest_start_time)):
                    # Get current rest start and end locations
                    current_rest_start = rest_start_loc[i]
                    current_rest_end = rest_end_loc[i]

                    new_eeg, new_timestamps = self.__interpolate(eeg_data[:, current_rest_start:current_rest_end], rest_timestamps, self.fsample)

                    rest_trials[i, :, :] = new_eeg
                    rest_timestamps = new_timestamps
                    
        except Exception:
            logger.warning("Failed to package resting state data")