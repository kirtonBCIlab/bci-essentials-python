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
from .paradigm import BaseParadigm
from .data_tank.data_tank import DataTank
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
        paradigm: BaseParadigm | None = None,
        data_tank: DataTank | None = None,
        messenger: Messenger | None = None,
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
        paradigm : BaseParadigm
            The paradigm used by EegData. This defines the processing and reshaping steps for the EEG data.
        data_tank : DataTank
            DataTank object to handle the storage of EEG trials and labels.  The default value is None.
        messenger: Messenger
            Messenger object to handle events from EegData, ex: acknowledging markers and
            predictions.  The default value is None.
        """

        # Ensure the incoming dependencies are the right type
        assert isinstance(classifier, GenericClassifier)
        assert isinstance(eeg_source, EegSource)
        assert isinstance(marker_source, MarkerSource | None)
        assert isinstance(paradigm, BaseParadigm | None)
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

        # Sort between event and command markers. Event markers are used by the data tank.
        # Command markers are used this controller currently called EegData.

        for marker in markers:
            marker = marker[0]
            if "Ping" in marker:
                continue

            # If the marker contains a single string, not including ',' and
            # begining with a alpha character, then it is an command marker
            marker_is_single_string = len(marker.split(",")) == 1
            marker_begins_with_alpha = marker[0].isalpha()
            is_command_marker = marker_is_single_string and marker_begins_with_alpha


            if is_command_marker:
                self.marker_data = np.append(self.marker_data, marker)
                self.marker_timestamps = np.append(self.marker_timestamps, timestamps[0])

            else:
                self.__data_tank.event_marker_strings = np.append(
                    self.__data_tank.event_marker_strings, marker
                )
                self.__data_tank.event_marker_timestamps = np.append(
                    self.__data_tank.event_marker_timestamps, timestamps[0]
                )

        print("debug")

        # # add the fresh data to the buffers
        # self.marker_data = np.concatenate((self.marker_data, markers))
        # self.marker_timestamps = np.concatenate((self.marker_timestamps, timestamps))

        # self.__data_tank.append_raw_markers(markers, timestamps)

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

        self.__data_tank.append_raw_eeg(eeg, timestamps)

        # Update latest EEG timestamp
        self.latest_eeg_timestamp = timestamps[-1]

    def setup(
        self,
        buffer_time=0.01,
        training=True,
        online=True,
        train_complete=False,
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

        Returns
        -------
        `None`

        """
        self.online = online
        self.training = training
        self.train_complete = train_complete

        self.buffer_time = buffer_time
        self.trial_end_buffer = buffer_time
        self.search_index = 0

        # initialize the numbers of markers and trials to zero
        self.marker_count = 0
        self.current_num_trials = 0
        self.n_trials = 0

        self.num_online_selections = 0
        self.online_selection_indices = []
        self.online_selections = []

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
        # read from sources to get new data. This puts command markers in the marker_data array and
        # event markers in the event_marker_strings array
        self._pull_data_from_sources()

        # check if there is an available command marker, if not, break and wait for more data
        while len(self.marker_timestamps) > self.marker_count:
            self.loops = 0

            # Get the current marker
            current_step_marker = self.marker_data[self.marker_count][0]

            if self._messenger is not None:
                # send feedback for each marker that you receive
                self._messenger.marker_received(current_step_marker)

            logger.info("Marker: %s", current_step_marker)

            # once all resting state data is collected then go and compile it
            # TODO
            if current_step_marker == "Done with all RS collection":
                self.__paradigm._package_resting_state_data(self.marker_data, self.marker_timestamps, self.eeg_data, self.eeg_timestamps)
                self.marker_count += 1

            elif current_step_marker == "Trial Started":
                logger.debug(
                    "Trial started, incrementing marker count and continuing"
                )
                # Note that a marker occured, but do nothing else
                self.marker_count += 1

            elif current_step_marker == "Trial Ends":
                # Tell the data tank to update the epoch array
                self.__data_tank.update_epochs()
                
                # Ask the paradigm if it needs to do anything
                

            elif current_step_marker == "Training Complete":
                # Pull the epochs from the data tank and pass them to the classifier
                X, y = self.__data_tank.get_training_data()
                self._classifier.add_to_train(X, y)
                self._classifier.fit()

            elif current_step_marker == "Update Classifier":
                # Pull the epochs from the data tank and pass them to the classifier
                X, y = self.__data_tank.get_training_data()
                self._classifier.add_to_train(X, y)
                self._classifier.fit()

            


                # # TRAIN
                # if self.training:
                #     self._classifier.add_to_train(
                #         self.current_processed_eeg_trials, self.current_labels
                #     )

                #     logger.debug(
                #         "%s trials and labels added to training set",
                #         self.current_num_trials,
                #     )

                #     # if iterative training is on and active then also make a prediction 
                #     if self.iterative_training:
                #         logger.info(
                #             "Added current samples to training set, "
                #             + "now making a prediction"
                #         )

                #         # Make a prediction
                #         prediction = self._classifier.predict(
                #             self.current_processed_eeg_trials
                #         )

                #         logger.info(
                #             "%s was selected by the iterative classifier",
                #             prediction.labels,
                #         )

                #         if self._messenger is not None:
                #             self._messenger.prediction(prediction)

                # # PREDICT
                # elif self.train_complete and self.current_num_trials != 0:
                #     logger.info(
                #         "Making a prediction based on %s trials",
                #         self.current_num_trials,
                #     )

                #     if self.current_num_trials == 0:
                #         logger.error("No trials to make a decision")
                #         self.marker_count += 1
                #         break

                #     # save the online selection indices
                #     selection_inds = list(
                #         range(
                #             self.n_trials - self.current_num_trials,
                #             self.n_trials,
                #         )
                #     )
                #     self.online_selection_indices.append(selection_inds)

                #     # make the prediciton
                #     try:
                #         prediction = self._classifier.predict(
                #             self.current_processed_eeg_trials
                #         )
                #         self.online_selections.append(prediction.labels)

                #         logger.info(
                #             "%s was selected by classifier", prediction.labels
                #         )

                #         if self._messenger is not None:
                #             self._messenger.prediction(prediction)

                #     except Exception:
                #         logger.warning("This classification failed...")

                # # OH DEAR
                # else:
                #     logger.error("Unable to classify... womp womp")

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