import numpy as np
from abc import ABC, abstractmethod

from ..utils.logger import Logger
from ..signal_processing import bandpass

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


class Paradigm(ABC):
    def __init__(self, filters=[5, 30], channel_subset=None):
        """
        Base class for all paradigms.
        Please use a subclass of this class for your specific paradigm.

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

        # When do we return classifications?
        self.classify_each_epoch = False
        self.classify_each_trial = False

        # Do we classify labeled epochs (such as in the case of iterative training)?
        self.classify_labeled_epochs = False

    def _preprocess(self, eeg, fsample, lowcut, highcut):
        """
        Preprocess EEG data with bandpass filter.

        Parameters
        ----------
        eeg : np.ndarray
            EEG data. Shape is (n_channels, n_samples).
        fsample : float
            Sampling frequency.
        lowcut : float
            Lower cutoff frequency.
        highcut : float
            Upper cutoff frequency.

        Returns
        -------
        np.ndarray
            Bandpassed EEG data. Shape is (n_channels, n_samples).

        """
        new_eeg = bandpass(eeg, lowcut, highcut, 5, fsample)

        return new_eeg

    def package_resting_state_data(
        self, marker_data, marker_timestamps, bci_controller, eeg_timestamps, fsample
    ):
        """Package resting state data.

        Parameters
        ----------
        marker_data : list of str
            List of markers.
        marker_timestamps : np.ndarray
            Timestamps of markers.
        bci_controller : np.ndarray
            EEG data. Shape is (n_channels, n_samples).
        eeg_timestamps : np.ndarray
            Timestamps of EEG data. Shape is (n_samples,).
        fsample : float
            Sampling frequency.

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

            self.fsample = fsample
            self.n_channels = bci_controller.shape[0]

            for i in range(len(marker_data)):
                # Get current resting state data marker and time stamp
                current_rs_data_marker = marker_data[i]
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

                new_eeg = np.zeros((1, self.n_channels, n_samples))

                eyes_open_timestamps = np.arange(0, duration, 1 / self.fsample)

                # Now copy EEG for these trials
                for i in range(len(eyes_open_start_time)):
                    eyes_open_time_correction = eyes_open_start_time[i]

                    corrected_eeg_timestamps = (
                        eeg_timestamps - eyes_open_time_correction
                    )

                    for c in range(self.n_channels):
                        new_eeg[0, c, :] = np.interp(
                            eyes_open_timestamps,
                            corrected_eeg_timestamps,
                            bci_controller[c, :],
                        )

                    if i == 0:
                        eyes_open_trials = new_eeg
                        eyes_open_timestamps = eyes_open_timestamps
                    else:
                        eyes_open_trials = np.concatenate(
                            (eyes_open_trials, new_eeg), axis=0
                        )

            # Eyes closed

            if len(eyes_closed_end_loc) > 0:
                # Get duration, nsmaples
                duration = np.floor(eyes_closed_end_time[0] - eyes_closed_start_time[0])
                n_samples = int(duration * self.fsample)

                new_eeg = np.zeros((1, self.n_channels, n_samples))

                eyes_closed_timestamps = np.arange(0, duration, 1 / self.fsample)

                # Now copy EEG for these trials
                for i in range(len(eyes_closed_start_time)):
                    eyes_closed_time_correction = eyes_closed_start_time[i]

                    corrected_eeg_timestamps = (
                        eeg_timestamps - eyes_closed_time_correction
                    )

                    for c in range(self.n_channels):
                        new_eeg[0, c, :] = np.interp(
                            eyes_closed_timestamps,
                            corrected_eeg_timestamps,
                            bci_controller[c, :],
                        )

                    if i == 0:
                        eyes_closed_trials = new_eeg
                        eyes_closed_timestamps = eyes_closed_timestamps
                    else:
                        eyes_closed_trials = np.concatenate(
                            (eyes_closed_trials, new_eeg), axis=0
                        )

            # Rest
            if len(rest_end_loc) > 0:
                # Get duration, nsmaples
                while rest_end_time[0] < rest_start_time[0]:
                    rest_end_time.pop(0)
                    rest_end_loc.pop(0)

                duration = np.floor(rest_end_time[0] - rest_start_time[0])
                n_samples = int(duration * self.fsample)

                new_eeg = np.zeros((1, self.n_channels, n_samples))

                rest_timestamps = np.arange(0, duration, 1 / self.fsample)

                # Now copy EEG for these trials
                for i in range(len(rest_start_time)):
                    rest_time_correction = rest_start_time[i]

                    corrected_eeg_timestamps = eeg_timestamps - rest_time_correction

                    for c in range(self.n_channels):
                        new_eeg[0, c, :] = np.interp(
                            rest_timestamps,
                            corrected_eeg_timestamps,
                            bci_controller[c, :],
                        )

                    if i == 0:
                        rest_trials = new_eeg
                        rest_timestamps = rest_timestamps

                    else:
                        rest_trials = np.concatenate((rest_trials, new_eeg), axis=0)

            # Put all the available data into a dictionary
            resting_state_data = {
                "eyes_open_trials": eyes_open_trials,
                "eyes_open_timestamps": eyes_open_timestamps,
                "eyes_closed_trials": eyes_closed_trials,
                "eyes_closed_timestamps": eyes_closed_timestamps,
                "rest_trials": rest_trials,
                "rest_timestamps": rest_timestamps,
            }

            logger.debug("Done packaging resting state data")

            return resting_state_data

        except Exception as e:
            logger.error(f"Error packaging resting state data: {e}")

            return None

    @abstractmethod
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

        pass

    @abstractmethod
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

        pass
