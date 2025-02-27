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
- `BciController` : For processing continuous data in trials of a defined
length.

"""

import time
import numpy as np

from .paradigm.paradigm import Paradigm
from .data_tank.data_tank import DataTank
from .classification.generic_classifier import GenericClassifier
from .io.sources import EegSource, MarkerSource
from .io.messenger import Messenger
from .utils.logger import Logger

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


# EEG data
class BciController:
    """
    Class that holds, trials, processes, and classifies EEG data.
    This class is used for processing of continuous EEG data in trials of a defined length.
    """

    def __init__(
        self,
        classifier: GenericClassifier,
        eeg_source: EegSource,
        marker_source: MarkerSource | None = None,
        paradigm: Paradigm | None = None,
        data_tank: DataTank | None = None,
        messenger: Messenger | None = None,
    ):
        """Initializes `BciController` class.

        Parameters
        ----------
        classifier : GenericClassifier
            The classifier used by BciController.
        eeg_source : EegSource
            Source of EEG data and timestamps, this could be from a file or headset via LSL, etc.
        marker_source : EegSource
            Source of Marker/Control data and timestamps, this could be from a file or Unity via
            LSL, etc.  The default value is None.
        paradigm : Paradigm
            The paradigm used by BciController. This defines the processing and reshaping steps for the EEG data.
        data_tank : DataTank
            DataTank object to handle the storage of EEG trials and labels.  The default value is None.
        messenger: Messenger
            Messenger object to handle events from BciController, ex: acknowledging markers and
            predictions.  The default value is None.
        """

        # Ensure the incoming dependencies are the right type
        assert isinstance(classifier, GenericClassifier)
        assert isinstance(eeg_source, EegSource)
        assert isinstance(marker_source, MarkerSource | None)
        assert isinstance(paradigm, Paradigm | None)
        assert isinstance(data_tank, DataTank | None)
        assert isinstance(messenger, Messenger | None)

        self._classifier = classifier
        self.__eeg_source = eeg_source
        self.__marker_source = marker_source
        self.__paradigm = paradigm
        self.__data_tank = data_tank
        self._messenger = messenger

        self.headset_string = self.__eeg_source.name
        self.fsample = self.__eeg_source.fsample
        self.n_channels = self.__eeg_source.n_channels
        self.ch_type = self.__eeg_source.channel_types
        self.ch_units = self.__eeg_source.channel_units
        self.channel_labels = self.__eeg_source.channel_labels

        self.__data_tank.set_source_data(
            self.headset_string,
            self.fsample,
            self.n_channels,
            self.ch_type,
            self.ch_units,
            self.channel_labels,
        )

        # Switch any trigger channels to stim, this is for mne/bids export (?)
        self.ch_type = [type.replace("trg", "stim") for type in self.ch_type]

        self._classifier.channel_labels = self.channel_labels

        logger.info(self.headset_string)
        logger.info(self.channel_labels)

        # Initialize data and timestamp arrays to the right dimensions, but zero elements
        self.marker_data = np.zeros((0, 1))
        self.marker_timestamps = np.zeros((0))
        self.bci_controller = np.zeros((0, self.n_channels))
        self.eeg_timestamps = np.zeros((0))

        self.ping_count = 0
        self.n_samples = 0
        self.time_units = ""

    # Get new data from source, whatever it is
    def _pull_data_from_sources(self):
        """Get pull data from EEG and optionally, the marker source.
        This method will fill up the marker_data, bci_controller and corresponding timestamp arrays.
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

        for i, marker in enumerate(markers):
            marker = marker[0]
            if "Ping" in marker:
                continue

            # Add all markers to the controller
            self.marker_data = np.append(self.marker_data, marker)
            self.marker_timestamps = np.append(self.marker_timestamps, timestamps[i])

            # Add all markers to the data tank
            self.__data_tank.add_raw_markers(
                np.array([marker]), np.array([timestamps[i]])
            )

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

        self.__data_tank.add_raw_eeg(eeg.T, timestamps)

        # Update latest EEG timestamp
        self.latest_eeg_timestamp = timestamps[-1]

    def __process_and_classify(self):
        """Process the markers and classify the data.

        Parameters
        ----------
            None

        Returns
        ----------
            success_flag : bool
                Flag indicating if the processing and classification was successful.

        """

        eeg_start_time, eeg_end_time = self.__paradigm.get_eeg_start_and_end_times(
            self.event_marker_buffer, self.event_timestamp_buffer
        )

        # No we actually need to wait until we have all the data for these markers
        eeg, timestamps = self.__data_tank.get_raw_eeg()

        # If the last timestamp is less than the end time, then we don't have the necessarty EEG to process
        if timestamps[-1] < eeg_end_time:
            return False

        X, y = self.__paradigm.process_markers(
            self.event_marker_buffer,
            self.event_timestamp_buffer,
            eeg,
            timestamps,
            self.fsample,
        )

        # Add the epochs to the data tank
        self.__data_tank.add_epochs(X, y)

        # If either there are no labels OR iterative training is on, then make a prediction
        if self.train_complete:
            if -1 in y or self.__paradigm.iterative_training:
                prediction = self._classifier.predict(X)
                self.__send_prediction(prediction)

        self.event_marker_buffer = []
        self.event_timestamp_buffer = []

        return True

    def __send_prediction(self, prediction):
        """Send a prediction to the messenger object."""
        if self._messenger is not None:
            self._messenger.prediction(prediction)

    def setup(
        self,
        online=True,
        train_complete=False,
        train_lock=False,
    ):
        """Configure processing loop.  This should be called before starting
        the loop with run() or step().  Calling after will reset the loop state.

        The processing loop reads in EEG and marker data and processes it.
        The loop can be run in "offline" or "online" modes:
        - If in `online` mode, then the loop will continuously try to read
        in data from the `BciController` object and process it. The loop will
        terminate when `max_loops` is reached, or when manually terminated.
        - If in `offline` mode, then the loop will read in all of the data
        at once, process it, and then terminate.

        Parameters
        ----------
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
        train_lock : bool, *optional*
            Flag to indicate if the classifier is locked (ie. no more training).
            - `True`: The classifier is locked.
            - `False`: The classifier is not locked.
            - Default is `False`.

        Returns
        -------
        `None`

        """
        self.online = online
        self.train_complete = train_complete
        self.train_lock = train_lock

        # initialize the numbers of markers and trials to zero
        self.marker_count = 0
        self.current_num_trials = 0
        self.n_trials = 0

        self.num_online_selections = 0
        self.online_selection_indices = []
        self.online_selections = []

    def run(self, max_loops: int = 1000000):
        """Runs BciController processing in a loop.
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
        else:
            self.loops = 0

        # Initialize the event marker buffer
        self.event_marker_buffer = []
        self.event_timestamp_buffer = []

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

    def step(self):
        """Runs a single BciController processing step.
        See setup() for configuration of processing.

        Parameters
        ----------
            None

        Returns
        ------
            None

        """
        # read from sources to get new data. This puts command markers in the marker_data array and
        # event markers in the event_marker_strings array
        self._pull_data_from_sources()

        # check if there is an available command marker, if not, break and wait for more data
        while len(self.marker_timestamps) > self.marker_count:
            # Get the current marker
            current_step_marker = self.marker_data[self.marker_count]
            current_timestamp = self.marker_timestamps[self.marker_count]

            if self._messenger is not None:
                # send feedback for each marker that you receive
                self._messenger.marker_received(current_step_marker)

            # If the marker contains a single string, then it is a command marker
            marker_is_single_string = len(current_step_marker.split(",")) == 1
            is_event_marker = not marker_is_single_string

            # Add the marker to the event marker buffer
            if is_event_marker:
                self.event_marker_buffer.append(current_step_marker)
                self.event_timestamp_buffer.append(current_timestamp)

                # If classification is on epochs, then update epochs, maybe classify, and clear the buffer
                if self.__paradigm.classify_each_epoch:
                    success_flag = self.__process_and_classify()
                    if success_flag is False:
                        break

            # TODO
            elif current_step_marker == "Done with all RS collection":
                (
                    self.bci_controller,
                    self.eeg_timestamps,
                ) = self.__data_tank.get_raw_eeg()

                resting_state_data = self.__paradigm.package_resting_state_data(
                    self.marker_data,
                    self.marker_timestamps,
                    self.bci_controller,
                    self.eeg_timestamps,
                    self.fsample,
                )

                self.__data_tank.add_resting_state_data(resting_state_data)

            elif current_step_marker == "Trial Started":
                logger.debug("Trial started, incrementing marker count and continuing")
                # Note that a marker occured, but do nothing else

            elif current_step_marker == "Trial Ends":
                # If we are classifying based on trials, then process the trial,
                if self.__paradigm.classify_each_trial:
                    success_flag = self.__process_and_classify()
                    if success_flag is False:
                        break

            elif current_step_marker == "Training Complete":
                if self.train_lock is False:
                    # Pull the epochs from the data tank and pass them to the classifier
                    X, y = self.__data_tank.get_epochs(latest=True)
                    if len(y) > 0:
                        self._classifier.add_to_train(X, y)
                    self._classifier.fit()
                    self.train_complete = True

            elif current_step_marker == "Update Classifier":
                if self.train_lock is False:
                    # Pull the epochs from the data tank and pass them to the classifier
                    X, y = self.__data_tank.get_epochs(latest=True)
                    if len(y) > 0:
                        self._classifier.add_to_train(X, y)
                    self._classifier.fit()
                    self.train_complete = True

            logger.info("Processed Marker: %s", current_step_marker)
            self.marker_count += 1
