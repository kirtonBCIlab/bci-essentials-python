"""Module for managing BCI data.

This module provides data classes for different BCI paradigms.

It includes the loading of offline data in `xdf` format
or the live streaming of LSL data.

The loaded/streamed data is added to a buffer such that offline and
online processing pipelines are identical.

Data is pre-processed (using the `signal_processing` module), divided into trials,
and classified (using one of the `classification` sub-modules).

Classes
-------
- `EegData` : For processing continuous data in trials of a defined
length.

"""

import time
import numpy as np

from .signal_processing import notch, bandpass
from .classification.generic_classifier import GenericClassifier
from .io.sources import EegSource, MarkerSource
from .io.messenger import Messenger
from .utils.logger import Logger

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


# EEG data
class EegData:
    """
    Class that holds, trials, processes, and classifies EEG data.
    This class is used for processing of continuous EEG data in trials of a defined length.
    """

    def __init__(
        self,
        classifier: GenericClassifier,
        eeg_source: EegSource,
        marker_source: MarkerSource | None = None,
        messenger: Messenger | None = None,
        subset: list[str] = [],
    ):
        """Initializes `EegData` class.

        Parameters
        ----------
        classifier : GenericClassifier
            The classifier used by EegData.
        eeg_source : EegSource
            Source of EEG data and timestamps, this could be from a file or headset via LSL, etc.
        marker_source : EegSource
            Source of Marker/Control data and timestamps, this could be from a file or Unity via
            LSL, etc.  The default value is None.
        messenger: Messenger
            Messenger object to handle events from EegData, ex: acknowledging markers and
            predictions.  The default value is None.
        subset : list of `int`, *optional*
            The list of EEG channel names to process, default is `[]`, meaning all channels.
        """

        # Ensure the incoming dependencies are the right type
        assert isinstance(classifier, GenericClassifier)
        assert isinstance(eeg_source, EegSource)
        assert isinstance(marker_source, MarkerSource | None)
        assert isinstance(messenger, Messenger | None)

        self._classifier = classifier
        self.__eeg_source = eeg_source
        self.__marker_source = marker_source
        self._messenger = messenger
        self.__subset = subset

        self.headset_string = self.__eeg_source.name
        self.fsample = self.__eeg_source.fsample
        self.n_channels = self.__eeg_source.n_channels
        self.ch_type = self.__eeg_source.channel_types
        self.ch_units = self.__eeg_source.channel_units
        self.channel_labels = self.__eeg_source.channel_labels

        # Switch any trigger channels to stim, this is for mne/bids export (?)
        self.ch_type = [type.replace("trg", "stim") for type in self.ch_type]

        # If a subset is to be used, define a new n_channels, channel labels, and eeg data
        if self.__subset != []:
            logger.info("A subset was defined")
            logger.info("Original channels\n%s", self.channel_labels)

            self.n_channels = len(self.__subset)
            self.subset_indices = []
            for s in self.__subset:
                self.subset_indices.append(self.channel_labels.index(s))

            self.channel_labels = self.__subset
            logger.info("Subset channels\n%s", self.channel_labels)

        else:
            self.subset_indices = list(range(0, self.n_channels))

        self._classifier.channel_labels = self.channel_labels

        logger.info(self.headset_string)
        logger.info(self.channel_labels)

        # Initialize data and timestamp arrays to the right dimensions, but zero elements
        self.marker_data = np.zeros((0, 1))
        self.marker_timestamps = np.zeros((0))
        self.eeg_data = np.zeros((0, self.n_channels))
        self.eeg_timestamps = np.zeros((0))

        self.ping_count = 0
        self.n_samples = 0
        self.time_units = ""

    def edit_settings(
        self,
        user_id="0000",
        n_channels=8,
        channel_labels=["?", "?", "?", "?", "?", "?", "?", "?"],
        fsample=256,
    ):
        """Override settings obtained from eeg_source on init

        Parameters
        ----------
        user_id : str, *optional*
            The user ID.
            - Default is `"0000"`.
        n_channels : int, *optional*
            The number of channels.
            - Default is `8`.
        channel_labels : list of `str`, *optional*
            The channel labels.
            - Default is `["?", "?", "?", "?", "?", "?", "?", "?"]`.
        fsample : int, *optional*
            The sampling rate.
            - Default is `256`.

        Returns
        -------
        `None`

        """
        self.user_id = user_id  # user id
        self.n_channels = n_channels  # number of channels
        self.channel_labels = channel_labels  # EEG electrode placements
        self.fsample = fsample  # sampling rate

        if len(channel_labels) != self.n_channels:
            logger.warning("Channel locations do not fit number of channels!!!")
            self.channel_labels = ["?"] * self.n_channels

    # Get new data from source, whatever it is
    def _pull_data_from_sources(self):
        """Get pull data from EEG and optionally, the marker source.
        This method will fill up the marker_data, eeg_data and corresponding timestamp arrays.
        """
        self.__pull_marker_data_from_source()
        self.__pull_eeg_data_from_source()

        # If the outlet exists send a ping
        if self._messenger is not None:
            self.ping_count += 1
            self._messenger.ping()

    def __pull_marker_data_from_source(self):
        """Pulls marker samples from source, sanity checks and appends to buffer"""

        # if there isn't a marker source, abort
        if self.__marker_source is None:
            return

        # read in the data
        markers, timestamps = self.__marker_source.get_markers()
        markers = np.array(markers)
        timestamps = np.array(timestamps)

        if markers.size == 0:
            return

        if markers.ndim != 2:
            logger.warning("discarded invalid marker data")
            return

        # apply time correction
        time_correction = self.__marker_source.time_correction()
        timestamps = [timestamps[i] + time_correction for i in range(len(timestamps))]

        # add the fresh data to the buffers
        self.marker_data = np.concatenate((self.marker_data, markers))
        self.marker_timestamps = np.concatenate((self.marker_timestamps, timestamps))

    def __pull_eeg_data_from_source(self):
        """Pulls eeg samples from source, sanity checks and appends to buffer"""

        # read in the data
        eeg, timestamps = self.__eeg_source.get_samples()
        eeg = np.array(eeg)
        timestamps = np.array(timestamps)

        if eeg.size == 0:
            return

        if eeg.ndim != 2:
            logger.warning("discarded invalid eeg data")
            return

        # handle subsets if needed
        if self.__subset != []:
            eeg = eeg[:, self.subset_indices]

        # if time is in milliseconds, divide by 1000, works for sampling rates above 10Hz
        try:
            if self.time_units == "milliseconds":
                timestamps = [(timestamps[i] / 1000) for i in range(len(timestamps))]

        # If time units are not defined then define them
        except Exception:
            dif_low = -2
            dif_high = -1
            while timestamps[dif_high] - timestamps[dif_low] == 0:
                dif_low -= 1
                dif_high -= 1

            if timestamps[dif_high] - timestamps[dif_low] > 0.1:
                timestamps = [(timestamps[i] / 1000) for i in range(len(timestamps))]
                self.time_units = "milliseconds"
            else:
                self.time_units = "seconds"

        # apply time correction, this is essential for headsets like neurosity which have their own clock
        time_correction = self.__eeg_source.time_correction()
        timestamps = [timestamps[i] + time_correction for i in range(len(timestamps))]

        # add the fresh data to the buffers
        self.eeg_data = np.concatenate((self.eeg_data, eeg))
        self.eeg_timestamps = np.concatenate((self.eeg_timestamps, timestamps))

    def save_trials_as_npz(self, file_name: str):
        """Saves EEG trials and labels as a numpy file.

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
        X = self.raw_eeg_trials
        y = self.labels

        # Cut X and y to be the lenght of the number of trials, because X and y are initialized to be the maximum number of trials
        X = X[: self.n_trials]
        y = y[: self.n_trials]

        # Save the raw EEG trials and labels as a numpy file
        np.savez(file_name, X=X, y=y)

    # SIGNAL PROCESSING
    # Preprocessing goes here (trials are n_channels by n_samples)
    def _preprocessing(self, trial, option=None, order=5, fc=60, fl=10, fh=50):
        """Signal preprocessing.

        Preprocesses the signal using one of the methods from the
        `signal_processing.py` module.

        Parameters
        ----------
        trial : numpy.ndarray
            Trial of EEG data.
            2D array containing data with `float` type.

            shape = (`n_channels`,`n_samples`)
        option : str, *optional*
            Preprocessing option. Options include:
            - `"notch"` : Notch filter
            - `"bandpass"` : Bandpass filter
            - Default is `None`.
        order : int, *optional*
            Order of the Bandpass filter.
            - Default is `5`.
        fc : int, *optional*
            Frequency of the notch filter.
            - Default is `60`.
        fl : int, *optional*
            Lower corner frequency of the bandpass filter.
            - Default is `10`.
        fh : int, *optional*
            Upper corner frequency of the bandpass filter.
            - Default is `50`.

        Returns
        -------
        new_trial : numpy.ndarray
            Preprocessed trial of EEG data.
            2D array containing data with `float` type.

            shape = (`n_channels`,`n_samples`)

        """
        # do nothing
        if option is None:
            new_trial = trial
            return new_trial

        if option == "notch":
            new_trial = notch(trial, fc=60, Q=30, fsample=self.fsample)
            return new_trial

        if option == "bandpass":
            new_trial = bandpass(trial, fl, fh, order, self.fsample)
            return new_trial

        # other preprocessing options go here

    # Artefact rejection goes here (trials are n_channels by n_samples)
    def _artefact_rejection(self, trial, option=None):
        """Artefact rejection.

        Parameters
        ----------
        trial : numpy.ndarray
            Trial of EEG data.
            2D array containing data with `float` type.

            shape = (`n_channels`,`n_samples`)
        option : str, *optional*
            Artefact rejection option. Options include:
            - Nothing has been implemented yet.
            - Default is `None`.

        Returns
        -------
        new_trial : numpy.ndarray
            Artefact rejected trial of EEG data.
            2D array containing data with `float` type.

            shape = (`n_channels`,`n_samples`)

        """
        # do nothing
        if option is None:
            new_trial = trial
            return new_trial

        # other preprocessing options go here\

    def _package_resting_state_data(self):
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

            current_time = self.eeg_timestamps[0]
            current_timestamp_loc = 0

            for i in range(len(self.marker_data)):
                # Get current resting state data marker and time stamp
                current_rs_data_marker = self.marker_data[i][0]
                current_rs_timestamp = self.marker_timestamps[i]

                # Increment the EEG until just past the marker timestamp
                while current_time < current_rs_timestamp:
                    current_timestamp_loc += 1
                    current_time = self.eeg_timestamps[current_timestamp_loc]

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
            # Get duration, nsmaples

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

                    # For each channel of the EEG, interpolate to uniform sampling rate
                    for c in range(self.n_channels):
                        # First, adjust the EEG timestamps to start from zero
                        eeg_timestamps_adjusted = (
                            self.eeg_timestamps[
                                current_eyes_open_start:current_eyes_open_end
                            ]
                            - self.eeg_timestamps[current_eyes_open_start]
                        )

                        # Second, interpolate to timestamps at a uniform sampling rate
                        channel_data = np.interp(
                            self.eyes_open_timestamps,
                            eeg_timestamps_adjusted,
                            self.eeg_data[
                                current_eyes_open_start:current_eyes_open_end, c
                            ],
                        )

                        # Third, add to the EEG trial
                        self.eyes_open_trials[i, c, :] = channel_data
                        self.eyes_open_timestamps

            logger.debug("Done packaging resting state data")

            # Eyes closed

            if len(eyes_closed_end_loc) > 0:
                # Get duration, nsmaples
                duration = np.floor(eyes_closed_end_time[0] - eyes_closed_start_time[0])
                n_samples = int(duration * self.fsample)

                self.eyes_closed_timestamps = np.array(range(n_samples)) / self.fsample
                self.eyes_closed_trials = np.ndarray(
                    (len(eyes_closed_start_time), self.n_channels, n_samples)
                )
                # Now copy EEG for these trials
                for i in range(len(eyes_closed_start_time)):
                    # Get current eyes closed start and end locations
                    current_eyes_closed_start = eyes_closed_start_loc[i]
                    current_eyes_closed_end = eyes_closed_end_loc[i]

                    # For each channel of the EEG, interpolate to uniform sampling rate
                    for c in range(self.n_channels):
                        # First, adjust the EEG timestamps to start from zero
                        eeg_timestamps_adjusted = (
                            self.eeg_timestamps[
                                current_eyes_closed_start:current_eyes_closed_end
                            ]
                            - self.eeg_timestamps[current_eyes_closed_start]
                        )

                        # Second, interpolate to timestamps at a uniform sampling rate
                        channel_data = np.interp(
                            self.eyes_closed_timestamps,
                            eeg_timestamps_adjusted,
                            self.eeg_data[
                                current_eyes_closed_start:current_eyes_closed_end, c
                            ],
                        )

                        # Third, add to the EEG trial
                        self.eyes_closed_trials[i, c, :] = channel_data
                        self.eyes_closed_timestamps

            # Rest
            if len(rest_end_loc) > 0:
                # Get duration, nsmaples
                while rest_end_time[0] < rest_start_time[0]:
                    rest_end_time.pop(0)
                    rest_end_loc.pop(0)

                duration = np.floor(rest_end_time[0] - rest_start_time[0])

                n_samples = int(duration * self.fsample)

                self.rest_timestamps = np.array(range(n_samples)) / self.fsample
                self.rest_trials = np.ndarray(
                    (len(rest_start_time), self.n_channels, n_samples)
                )
                # Now copy EEG for these trials
                for i in range(len(rest_start_time)):
                    # Get current rest start and end locations
                    current_rest_start = rest_start_loc[i]
                    current_rest_end = rest_end_loc[i]

                    # For each channel of the EEG, interpolate to uniform sampling rate
                    for c in range(self.n_channels):
                        # First, adjust the EEG timestamps to start from zero
                        eeg_timestamps_adjusted = (
                            self.eeg_timestamps[current_rest_start:current_rest_end]
                            - self.eeg_timestamps[current_rest_start]
                        )

                        # Second, interpolate to timestamps at a uniform sampling rate
                        channel_data = np.interp(
                            self.rest_timestamps,
                            eeg_timestamps_adjusted,
                            self.eeg_data[current_rest_start:current_rest_end, c],
                        )

                        # Third, add to the EEG trial
                        self.rest_trials[i, c, :] = channel_data
                        self.rest_timestamps
        except Exception:
            logger.warning("Failed to package resting state data")

    def setup(
        self,
        buffer_time=0.01,
        max_channels=64,
        max_samples=2560,
        max_trials=1000,
        training=True,
        online=True,
        train_complete=False,
        iterative_training=False,
        live_update=False,
        pp_type="bandpass",  # preprocessing method
        pp_low=1,  # bandpass lower cutoff
        pp_high=40,  # bandpass upper cutoff
        pp_order=5,  # bandpass order
    ):
        """Configure processing loop.  This should be called before starting
        the loop with run() or step().  Calling after will reset the loop state.

        The processing loop reads in EEG and marker data and processes it.
        The loop can be run in "offline" or "online" modes:
        - If in `online` mode, then the loop will continuously try to read
        in data from the `EegData` object and process it. The loop will
        terminate when `max_loops` is reached, or when manually terminated.
        - If in `offline` mode, then the loop will read in all of the data
        at once, process it, and then terminate.

        Parameters
        ----------
        buffer_time : float, *optional*
            Buffer time for EEG sampling in `online` mode (seconds).
            - Default is `0.01`.
        max_channels : int, *optional*
            Maximum number of EEG channels to read in.
            - Default is `64`.
        max_samples : int, *optional*
            Maximum number of EEG samples to read in per trial.
            - Default is `2560`.
        max_trials : int, *optional*
            Maximum number of trials to read in per loop (?).
            - Default is `1000`.
        max_loops : int, *optional*
            Maximum number of loops to run.
            - Default is `1000000`.
        training : bool, *optional*
            Flag to indicate if the data will be used to train a classifier.
            - `True`: The data will be used to train the classifier.
            - `False`: The data will be used to predict with the classifier.
            - Default is `True`.
        online : bool, *optional*
            Flag to indicate if the data will be processed in `online` mode.
            - `True`: The data will be processed in `online` mode.
            - `False`: The data will be processed in `offline` mode.
            - Default is `True`.
        train_complete : bool, *optional*
            Flag to indicate if the classifier has been trained.
            - `True`: The classifier has been trained.
            - `False`: The classifier has not been trained.
            - Default is `False`.
        iterative_training : bool, *optional*
            Flag to indicate if the classifier will be updated iteratively.
            - Default is `False`.
        live_update : bool, *optional*
            Flag to indicate if the classifier will be used to provide
            live updates on trial classification.
            - Default is `False`.
        pp_type : str, *optional*
            Preprocessing method to apply to the EEG data.
            - Default is `"bandpass"`.
        pp_low : int, *optional*
            Low corner frequency for bandpass filter.
            - Default is `1`.
        pp_high : int, *optional*
            Upper corner frequency for bandpass filter.
            - Default is `40`.
        pp_order : int, *optional*
            Order of the bandpass filter.
            - Default is `5`.

        Returns
        -------
        `None`

        """
        self.online = online
        self.training = training
        self.live_update = live_update
        self.iterative_training = iterative_training
        self.train_complete = train_complete

        self.max_channels = max_channels
        self.max_samples = max_samples
        self.max_trials = max_trials

        self.pp_type = pp_type
        self.pp_low = pp_low
        self.pp_high = pp_high
        self.pp_order = pp_order

        self.buffer_time = buffer_time
        self.trial_end_buffer = buffer_time
        self.search_index = 0

        # initialize trials and labels
        self.current_raw_eeg_trials = np.zeros(
            (self.max_trials, self.max_channels, self.max_samples)
        )
        self.current_processed_eeg_trials = self.current_raw_eeg_trials
        self.current_labels = np.zeros((self.max_trials))

        self.raw_eeg_trials = np.zeros(
            (self.max_trials, self.max_channels, self.max_samples)
        )
        self.processed_eeg_trials = self.raw_eeg_trials
        self.labels = np.zeros((self.max_trials))  # temporary labels
        self.training_labels = np.zeros((self.max_trials))  # permanent training labels

        # initialize the numbers of markers and trials to zero
        self.marker_count = 0
        self.current_num_trials = 0
        self.n_trials = 0

        self.num_online_selections = 0
        self.online_selection_indices = []
        self.online_selections = []

        # initialize loop count, TODO - why do this here?
        self.loops = 0

    def run(self, max_loops: int = 1000000):
        """Runs EegData processing in a loop.
        See setup() for configuration of processing.

        Parameters
        ----------
        max_loops : int, *optional*
            Maximum number of loops to run, default is `1000000`.

        Returns
        ------
            None

        """
        # if offline, then all data is already loaded, only need to loop once
        if self.online is False:
            self.loops = max_loops - 1

        # start the main loop, stops after pulling new data, max_loops times
        while self.loops < max_loops:
            # print out loop status
            if self.loops % 100 == 0:
                logger.debug(self.loops)

            if self.loops == max_loops - 1:
                logger.debug("last loop")

            # read from sources and process
            self.step()

            # Wait a short period of time and then try to pull more data
            if self.online:
                time.sleep(0.00001)

            self.loops += 1

        # Trim all the data
        self.raw_eeg_trials = self.raw_eeg_trials[
            0 : self.n_trials, 0 : self.n_channels, 0 : self.n_samples
        ]
        self.processed_eeg_trials = self.processed_eeg_trials[
            0 : self.n_trials, 0 : self.n_channels, 0 : self.n_samples
        ]
        self.labels = self.labels[0 : self.n_trials]

    def step(self):
        """Runs a single EegData processing step.
        See setup() for configuration of processing.

        Parameters
        ----------
        max_loops : int, *optional*
            Maximum number of loops to run, default is `1000000`.

        Returns
        ------
            None

        """
        # read from sources to get new data
        self._pull_data_from_sources()

        # check if there is an available marker, if not, break and wait for more data
        while len(self.marker_timestamps) > self.marker_count:
            self.loops = 0

            # Get the current marker
            current_step_marker = self.marker_data[self.marker_count][0]

            # If the marker contains a single string, not including ',' and
            # begining with a alpha character, then it is an event message
            marker_is_single_string = len(current_step_marker.split(",")) == 1
            marker_begins_with_alpha = current_step_marker[0].isalpha()
            is_event_marker = marker_is_single_string and marker_begins_with_alpha

            if is_event_marker:
                if self._messenger is not None:
                    # send feedback for each marker that you receive
                    self._messenger.marker_received(current_step_marker)

                logger.info("Marker: %s", current_step_marker)

                # once all resting state data is collected then go and compile it
                if current_step_marker == "Done with all RS collection":
                    self._package_resting_state_data()
                    self.marker_count += 1

                elif current_step_marker == "Trial Started":
                    logger.debug(
                        "Trial started, incrementing marker count and continuing"
                    )
                    # Note that a marker occured, but do nothing else
                    self.marker_count += 1

                elif current_step_marker == "Trial Ends":
                    logger.debug("Trial ended, trim the unused ends of numpy arrays")
                    # Trim the unused ends of numpy arrays
                    self.current_raw_eeg_trials = self.current_raw_eeg_trials[
                        0 : self.current_num_trials,
                        0 : self.n_channels,
                        0 : self.n_samples,
                    ]
                    self.current_processed_eeg_trials = (
                        self.current_processed_eeg_trials[
                            0 : self.current_num_trials,
                            0 : self.n_channels,
                            0 : self.n_samples,
                        ]
                    )
                    self.current_labels = self.current_labels[
                        0 : self.current_num_trials
                    ]

                    # TRAIN
                    if self.training:
                        self._classifier.add_to_train(
                            self.current_processed_eeg_trials, self.current_labels
                        )

                        logger.debug(
                            "%s trials and labels added to training set",
                            self.current_num_trials,
                        )

                        # if iterative training is on and active then also make a prediction
                        if self.iterative_training:
                            logger.info(
                                "Added current samples to training set, "
                                + "now making a prediction"
                            )

                            # Make a prediction
                            prediction = self._classifier.predict(
                                self.current_processed_eeg_trials
                            )

                            logger.info(
                                "%s was selected by the iterative classifier",
                                prediction.labels,
                            )

                            if self._messenger is not None:
                                self._messenger.prediction(prediction)

                    # PREDICT
                    elif self.train_complete and self.current_num_trials != 0:
                        logger.info(
                            "Making a prediction based on %s trials",
                            self.current_num_trials,
                        )

                        if self.current_num_trials == 0:
                            logger.error("No trials to make a decision")
                            self.marker_count += 1
                            break

                        # save the online selection indices
                        selection_inds = list(
                            range(
                                self.n_trials - self.current_num_trials,
                                self.n_trials,
                            )
                        )
                        self.online_selection_indices.append(selection_inds)

                        # make the prediciton
                        try:
                            prediction = self._classifier.predict(
                                self.current_processed_eeg_trials
                            )
                            self.online_selections.append(prediction.labels)

                            logger.info(
                                "%s was selected by classifier", prediction.labels
                            )

                            if self._messenger is not None:
                                self._messenger.prediction(prediction)

                        except Exception:
                            logger.warning("This classification failed...")

                    # OH DEAR
                    else:
                        logger.error("Unable to classify... womp womp")

                    # Reset trials and labels
                    self.marker_count += 1
                    self.current_num_trials = 0
                    self.current_raw_eeg_trials = np.zeros(
                        (self.max_trials, self.max_channels, self.max_samples)
                    )
                    self.current_processed_eeg_trials = self.current_raw_eeg_trials
                    self.current_labels = np.zeros((self.max_trials))

                # If human training completed then train the classifier
                elif (
                    current_step_marker == "Training Complete"
                    and self.train_complete is False
                ):
                    logger.debug("Training the classifier")

                    self._classifier.fit()
                    self.train_complete = True
                    self.training = False
                    self.marker_count += 1

                elif current_step_marker == "Update Classifier":
                    logger.debug("Retraining the classifier")

                    self._classifier.fit()

                    self.iterative_training = True
                    if self.online:
                        self.live_update = True

                    self.marker_count += 1

                else:
                    self.marker_count += 1

                if self.online:
                    time.sleep(0.01)
                self.loops += 1
                continue

            # Get marker info
            current_marker_info = current_step_marker.split(",")

            self.paradigm_string = current_marker_info[0]
            self.num_options = int(current_marker_info[1])
            label = int(current_marker_info[2])
            self.trial_length = float(current_marker_info[3])  # trial length
            if (
                len(current_marker_info) > 4
            ):  # if longer, collect this info and maybe it can be used by the classifier
                self.meta = []
                for i in range(4, len(current_marker_info)):
                    self.meta.__add__([current_marker_info[i]])

                # Load the correct SSVEP freqs
                if current_marker_info[0] == "ssvep":
                    self._classifier.target_freqs = [1] * (len(current_marker_info) - 4)
                    self._classifier.sampling_freq = self.fsample
                    for i in range(4, len(current_marker_info)):
                        self._classifier.target_freqs[i - 4] = float(
                            current_marker_info[i]
                        )
                        logger.debug(
                            "Changed %s target frequency to %s",
                            i - 4,
                            current_marker_info[i],
                        )

            # Check if the whole EEG trial corresponding to the marker is available
            end_time_plus_buffer = (
                self.marker_timestamps[self.marker_count]
                + self.trial_length
                + self.buffer_time
            )

            # If we don't have the full trial then pull more data, only do this online
            if self.eeg_timestamps[-1] <= end_time_plus_buffer:
                if self.online is True:
                    break
                if self.online is False:
                    self.marker_count += 1
                break

            logger.info("Marker information: %s", current_marker_info)

            # send message if there is an available outlet
            if self._messenger is not None:
                logger.info("sending marker back")
                # send feedback for each marker that you receive
                self._messenger.marker_received(current_step_marker)

            # Find the start time for the trial based on the marker timestamp
            start_time = self.marker_timestamps[self.marker_count]

            # Find the number of samples per trial
            self.n_samples = int(self.trial_length * self.fsample)

            # set the trial timestamps at exactly the sampling frequency
            self.trial_timestamps = np.arange(self.n_samples) / self.fsample

            # locate the indices of the trial in the eeg data
            for i, s in enumerate(self.eeg_timestamps[self.search_index : -1]):
                if s > start_time:
                    start_loc = self.search_index + i - 1
                    break

            # Get the end location for the trial
            end_loc = int(start_loc + self.n_samples + 1)

            # For each channel of the EEG, interpolate to uniform sampling rate
            for c in range(self.n_channels):
                # First, adjust the EEG timestamps to start from zero
                eeg_timestamps_adjusted = (
                    self.eeg_timestamps[start_loc:end_loc]
                    - self.eeg_timestamps[start_loc]
                )

                # Second, interpolate to timestamps at a uniform sampling rate
                channel_data = np.interp(
                    self.trial_timestamps,
                    eeg_timestamps_adjusted,
                    self.eeg_data[start_loc:end_loc, c],
                )

                # Third, sdd to the EEG trial
                self.current_raw_eeg_trials[
                    self.current_num_trials, c, 0 : self.n_samples
                ] = channel_data

            # This is where to do preprocessing
            self.current_processed_eeg_trials[
                self.current_num_trials, : self.n_channels, : self.n_samples
            ] = self._preprocessing(
                trial=self.current_raw_eeg_trials[
                    self.current_num_trials, : self.n_channels, : self.n_samples
                ],
                option=self.pp_type,
                order=self.pp_order,
                fl=self.pp_low,
                fh=self.pp_high,
            )

            # This is where to do artefact rejection
            self.current_processed_eeg_trials[
                self.current_num_trials, : self.n_channels, : self.n_samples
            ] = self._artefact_rejection(
                trial=self.current_processed_eeg_trials[
                    self.current_num_trials, : self.n_channels, : self.n_samples
                ],
                option=None,
            )

            # Add the label if it exists, otherwise set a flag of -1 to denote that there is no label
            # if self.training:
            #     self.current_labels[self.current_num_trials] = label
            # else:
            #     self.current_labels[self.current_num_trials] = -1
            self.current_labels[self.current_num_trials] = label

            # copy to the eeg_data object
            self.raw_eeg_trials[
                self.n_trials, 0 : self.n_channels, 0 : self.n_samples
            ] = self.current_raw_eeg_trials[
                self.current_num_trials, 0 : self.n_channels, 0 : self.n_samples
            ]
            self.processed_eeg_trials[
                self.n_trials, 0 : self.n_channels, 0 : self.n_samples
            ] = self.current_processed_eeg_trials[
                self.current_num_trials, 0 : self.n_channels, 0 : self.n_samples
            ]
            self.labels[self.n_trials] = self.current_labels[self.current_num_trials]

            # Send live updates
            if self.live_update:
                try:
                    if self.n_samples != 0:
                        prediction = self._classifier.predict(
                            self.current_processed_eeg_trials[
                                self.current_num_trials,
                                0 : self.n_channels,
                                0 : self.n_samples,
                            ]
                        )
                        self._messenger.prediction(prediction)
                except Exception:
                    logger.error("Unable to classify this trial")

            # iterate to next trial
            self.marker_count += 1
            self.current_num_trials += 1
            self.n_trials += 1
            self.search_index = start_loc
