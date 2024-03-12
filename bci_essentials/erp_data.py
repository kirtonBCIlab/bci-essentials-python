"""Module for managing ERP data.

This module provides data classes for different BCI paradigms.

It includes the loading of offline data in `xdf` format
or the live streaming of LSL data.

The loaded/streamed data is added to a buffer such that offline and
online processing pipelines are identical.

Data is pre-processed (using the `signal_processing` module), separated into trials,
and classified (using one of the `classification` sub-modules).

Classes
-------
- `ErpData` : For processing P300 or other Event Related Potentials
(ERP).

"""

import time
import numpy as np
import matplotlib.pyplot as plt

from .eeg_data import EegData
from .utils.logger import Logger

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


# ERP Data
class ErpData(EegData):
    """
    Class that holds, trials, processes and classifies ERP data.
    """

    # Formats the ERP data, call this every time that a new chunk arrives
    def setup(
        self,
        trial_start=0.0,
        trial_end=0.8,
        buffer_time=0.01,
        max_num_options=64,
        max_trials_per_option=50,
        max_trials=10000,
        max_decisions=500,
        training=False,
        train_complete=False,
        online=False,
        pp_type="bandpass",  # Preprocessing method
        pp_low=1,  # Preprocessing: bandpass lower cutoff
        pp_high=40,  # Preprocessing: bandpass upper cutoff
        pp_order=5,  # Preprocessing: bandpass order
        plot_erp=False,
    ):
        """Runs ErpData processing in a loop.

        Formats the ERP data. Call this every time that a new chunk arrives.

        Runs a while loop that reads in ERP trials. The loop can be run in
        "offline" or "online" modes:
        - If in `online` mode, then the loop will continuously try to read
        in data from the `EegData` object and process it. The loop will
        terminate when `max_loops` is reached, or when manually terminated.
        - If in `offline` mode, then the loop will read in all of the data
        at once, process it, and then terminate.

        Parameters
        ----------
        trial_start : float, *optional*
            Start time for ERP sampling trial relative to marker (seconds).
            - Default is `0.0`.
        trial_end : float, *optional*
            End time for ERP sampling trial relative to marker (seconds).
            - Default is `0.8`.
        buffer : float, *optional*
            Buffer time for EEG sampling in `online` mode (seconds).
            - Default is `0.01`.
        max_num_options : int, *optional*
            Maximum number of stimulus options (?).
            - Default is `64`.
        max_trials_per_option : int, *optional*
            Maximum number of trials to read in per stimulus option (?).
            - Default is `50`.
        max_trials : int, *optional*
            Maximum number of trials to read in per loop (?).
            - Default is `1000`.
        max_decisions : int, *optional*
            Maximum number of ERP decision blocks to store per loop (?).
            - Default is `500`.
        max_loops : int, *optional*
            Maximum number of loops to run.
            - Default is `1000000`.
        training : bool, *optional*
            Flag to indicate if the data will be used to train a classifier.
            - `True`: The data will be used to train the classifier.
            - `False`: The data will be used to predict with the classifier.
            - Default is `True`.
        train_complete : bool, *optional*
            Flag to indicate if the classifier has been trained.
            - `True`: The classifier has been trained.
            - `False`: The classifier has not been trained.
            - Default is `False`.
        online : bool, *optional*
            Flag to indicate if the data will be processed in `online` mode.
            - `True`: The data will be processed in `online` mode.
            - `False`: The data will be processed in `offline` mode.
            - Default is `True`.
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
        plot_erp : bool, *optional*
            - Default is `False`.

        Returns
        -------
        `None`

        """

        self.trial_start = trial_start
        self.trial_end = trial_end
        self.buffer_time = buffer_time

        self.training = training
        self.train_complete = train_complete
        self.online = online

        self.pp_type = pp_type
        self.pp_low = pp_low
        self.pp_high = pp_high
        self.pp_order = pp_order

        self.unity_train = True
        self.unity_label = "null"

        # plot settings
        self.plot_erp = plot_erp
        if self.plot_erp:
            self.fig1, self.axs1 = plt.subplots(self.n_channels)
            self.fig2, self.axs2 = plt.subplots(self.n_channels)
            self.non_target_plot = 99

        # iff this is the first time this function is being called for a given dataset
        self.trial_size = self.trial_end - self.trial_start
        self.n_samples = int(np.ceil(self.trial_size * self.fsample) + 1)
        self.trial_end_buffer = self.buffer_time
        self.num_options = max_num_options
        self.max_trials = max_trials
        self.max_trials_per_option = max_trials_per_option
        self.max_decisions = max_decisions

        self.search_index = 0

        self.trial_timestamps = np.arange(self.n_samples) / self.fsample

        # initialize the numbers of markers, trials, and decision blocks to zero
        self.marker_count = 0
        self.n_trials = 0
        self.decision_count = 0

        self.training_labels = np.zeros((self.max_trials), dtype=int)
        self.stim_labels = np.zeros((self.max_trials, self.num_options), dtype=bool)
        self.target_index = np.ndarray((self.max_trials), bool)

        # initialize the data structures in numpy arrays
        # ERP trials
        self.erp_trials_raw = np.zeros(
            (self.max_trials, self.n_channels, self.n_samples)
        )
        self.erp_trials_processed = np.zeros(
            (self.max_trials, self.n_channels, self.n_samples)
        )

        # Trials per decision, ie. the number of times each stimulus has flashed
        self.trials_per_decision = np.zeros((self.num_options))

        # Decision blocks are the ensemble averages of all trials collected for each stimulus object
        self.decision_blocks_raw = np.ndarray(
            (self.max_decisions, self.num_options, self.n_channels, self.n_samples)
        )
        self.decision_blocks_processed = np.ndarray(
            (self.max_decisions, self.num_options, self.n_channels, self.n_samples)
        )

        # Big decision blocks contain all decisions, all stimulus objects, all trials, all channels, and all samples (they are BIG)
        self.big_decision_blocks_raw = np.ndarray(
            (
                self.max_decisions,
                self.num_options,
                self.max_trials_per_option,
                self.n_channels,
                self.n_samples,
            )
        )
        self.big_decision_blocks_processed = np.ndarray(
            (
                self.max_decisions,
                self.num_options,
                self.max_trials_per_option,
                self.n_channels,
                self.n_samples,
            )
        )

        self.num_options_per_decision = np.zeros((max_decisions))

        self.loops = 0

    def run(self, max_loops=1000000000):
        """Runs ErpData processing in a loop.
        See setup() for configuration of processing.

        Parameters
        ----------
        max_loops : int, *optional*
            Maximum number of loops to run, default is `1000000`.

        Returns
        ------
            None

        """

        while self.loops < max_loops:
            # load data chunk from search start position
            # offline no more data to load
            if self.online is False:
                self.loops = max_loops

            # read from sources and process
            self.step()

            if self.online:
                time.sleep(0.000001)
            self.loops += 1

        # Trim the unused ends of numpy arrays
        if self.training:
            self.training_labels = self.training_labels[0 : self.n_trials - 1]
            self.target_index = self.target_index[0 : self.n_trials - 1]

        # self.erp_trials = self.erp_trials[0:self.n_trials, 0:self.n_channels, 0:self.n_samples]
        self.erp_trials_raw = self.erp_trials_raw[
            0 : self.n_trials, 0 : self.n_channels, 0 : self.n_samples
        ]
        self.target_index = self.target_index[0 : self.n_trials]
        self.training_labels = self.training_labels[0 : self.n_trials]
        self.stim_labels = self.stim_labels[0 : self.n_trials, :]
        self.num_options_per_decision = self.num_options_per_decision[
            0 : self.decision_count
        ]
        self.decision_blocks_raw = self.decision_blocks_raw[
            0 : self.decision_count, :, 0 : self.n_channels, 0 : self.n_samples
        ]
        self.decision_blocks_processed = self.decision_blocks_processed[
            0 : self.decision_count, :, 0 : self.n_channels, 0 : self.n_samples
        ]
        self.big_decision_blocks_raw = self.big_decision_blocks_raw[
            0 : self.decision_count, :, :, 0 : self.n_channels, 0 : self.n_samples
        ]
        self.big_decision_blocks_processed = self.big_decision_blocks_processed[
            0 : self.decision_count, :, :, 0 : self.n_channels, 0 : self.n_samples
        ]

    def step(self):
        """Runs a single ErpData processing step.
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
            if len(current_step_marker.split(",")) == 1:
                if (
                    current_step_marker == "P300 SingleFlash Started"
                    or current_step_marker == "P300 SingleFlash Begins"
                    or current_step_marker == "Trial Started"
                ):
                    # Note that a marker occured, but do nothing else
                    logger.info("Trial Started")
                    self.marker_count += 1
                    logger.debug("Increased marker count by 1 to %s", self.marker_count)

                # once all resting state data is collected then go and compile it
                elif current_step_marker == "Done with all RS collection":
                    self._package_resting_state_data()
                    self.marker_count += 1

                # If training completed then train the classifier
                elif (
                    current_step_marker == "Training Complete"
                    and self.train_complete is False
                ):
                    if self.train_complete is False:
                        logger.debug("Training the classifier")
                        self._classifier.fit()
                    self.train_complete = True
                    self.training = False
                    self.marker_count += 1
                    # continue

                # if there is a P300 end flag increment the decision_index by one
                elif (
                    current_step_marker == "P300 SingleFlash Ends"
                    or current_step_marker == "Trial Ends"
                ):
                    # get the smallest number of trials per decision in the case the are not the same
                    num_ensemble_trials = int(np.min(self.trials_per_decision))

                    # save the number of options
                    self.num_options_per_decision[self.decision_count] = int(
                        self.num_options
                    )

                    # Raw ensemble average
                    ensemble_average_block = np.mean(
                        self.big_decision_blocks_raw[
                            self.decision_count,
                            0 : self.num_options,
                            0:num_ensemble_trials,
                            0 : self.n_channels,
                            0 : self.n_samples,
                        ],
                        axis=1,
                    )
                    self.decision_blocks_raw[
                        self.decision_count,
                        0 : self.num_options,
                        0 : self.n_channels,
                        0 : self.n_samples,
                    ] = ensemble_average_block

                    # Processed ensemble average
                    ensemble_average_block = np.mean(
                        self.big_decision_blocks_processed[
                            self.decision_count,
                            0 : self.num_options,
                            0:num_ensemble_trials,
                            0 : self.n_channels,
                            0 : self.n_samples,
                        ],
                        axis=1,
                    )
                    self.decision_blocks_processed[
                        self.decision_count,
                        0 : self.num_options,
                        0 : self.n_channels,
                        0 : self.n_samples,
                    ] = ensemble_average_block

                    # Reset trials per decision
                    self.trials_per_decision = np.zeros((self.num_options))

                    if self.plot_erp:
                        self.fig1.show()
                        self.fig2.show()

                    logger.info("Marker: %s", current_step_marker)

                    self.marker_count += 1

                    # ADD IN A CHECK TO MAKE SURE THERE IS SUFFICIENT INFO TO MAKE A DECISION
                    # IF NOT THEN MOVE ON TO THE NEXT ONE
                    if True:
                        # CLASSIFICATION
                        # if the decision block has a label then add to training set
                        # if training:
                        # if self.decision_count <= len(self.labels) - 1:
                        if self.train_complete is False:
                            # ADD to training set
                            if self.unity_train:
                                logger.debug(
                                    "Adding decision block %s to "
                                    + "the classifier with label %s",
                                    self.decision_count,
                                    self.unity_label,
                                )
                                self._classifier.add_to_train(
                                    self.decision_blocks_processed[
                                        self.decision_count,
                                        : self.num_options,
                                        :,
                                        :,
                                    ],
                                    self.unity_label,
                                )

                                # plot what was added
                                # decision_vis(self.decision_blocks[self.decision_count,:,:,:], self.fsample, unity_label, self.channel_labels)
                            else:
                                logger.debug(
                                    "Adding decision block %s to "
                                    + "the classifier with label %s",
                                    self.decision_count,
                                    self.labels[self.decision_count],
                                )
                                self._classifier.add_to_train(
                                    self.decision_blocks_processed[
                                        self.decision_count,
                                        : self.num_options,
                                        :,
                                        :,
                                    ],
                                    self.labels[self.decision_count],
                                )

                                # if the last of the labelled data was just added
                                if self.decision_count == len(self.labels) - 1:
                                    # FIT
                                    logger.debug("Training the classifier")
                                    self._classifier.fit(n_splits=len(self.labels))

                        # else do the predict the label
                        else:
                            # PREDICT
                            prediction = self._classifier.predict_decision_block(
                                decision_block=self.decision_blocks_processed[
                                    self.decision_count, 0 : self.num_options, :, :
                                ],
                            )

                            logger.info(
                                "%s was selected by the classifier", prediction.labels
                            )

                            if self._messenger is not None:
                                logger.info("Sending prediction %s", prediction)
                                self._messenger.prediction(prediction)

                    # TODO: Code is currently unreachable
                    else:
                        logger.error("Insufficient trials to make a decision")
                        self.decision_count -= 1
                        # logger.info(
                        #     "Trials per decision: %s",
                        #     self.trials_per_decision
                        # )

                    self.decision_count += 1
                    self.trials_per_decision = np.zeros((self.num_options))

                    # UPDATE THE SEARCH START LOC
                    # continue
                else:
                    self.marker_count += 1

                if self.online:
                    time.sleep(0.01)
                self.loops += 1
                continue

            # Check if the whole EEG trial corresponding to the marker is available
            end_time_plus_buffer = (
                self.marker_timestamps[self.marker_count]
                + self.trial_end
                + self.buffer_time
            )

            if self.eeg_timestamps[-1] <= end_time_plus_buffer:
                # UPDATE THE SEARCH START LOC
                break

            if self._messenger is not None:
                self._messenger.marker_received(current_step_marker)

            # Get marker info
            current_marker_info = current_step_marker.split(",")

            # unity_flash_indexes
            flash_indices = list()

            for i, info in enumerate(current_marker_info):
                # if i == 0:
                #     bci_string = info
                if i == 1:
                    self.flash_type = info
                elif i == 2:
                    # If there is a different number of options
                    if self.num_options != int(info):
                        self.num_options = int(info)

                        # Resize on the first marker
                        self.trials_per_decision = np.zeros((self.num_options))
                elif i == 3:
                    self.unity_label = int(info)
                elif i >= 4:
                    flash_indices.append(int(info))

            self.trials_per_decision[flash_indices] += 1

            # During training,
            # should this be repeated for multiple flash indices
            # for flash_index in flash_indices:
            if self.training:
                # Get target info
                # current_target = target_order[self.decision_count]
                if self.unity_train:
                    logger.info("Marker information: %s", current_marker_info)
                    current_target = self.unity_label

                self.training_labels[self.n_trials] = current_target

                for fi in flash_indices:
                    self.stim_labels[self.n_trials, fi] = True

                if current_target in flash_indices:
                    self.target_index[self.n_trials] = True
                else:
                    self.target_index[self.n_trials] = False

            # Find the start time and end time for the trial based on the marker timestamp
            start_time = self.marker_timestamps[self.marker_count] + self.trial_start
            # end_time = self.marker_timestamps[self.marker_count] + self.trial_end

            # locate the indices of the trial in the eeg data
            for i, s in enumerate(self.eeg_timestamps[self.search_index : -1]):
                logger.debug(
                    "Indices (i,s) of the trial in the eeg data: (%s,%s)", i, s
                )
                if s > start_time:
                    start_loc = self.search_index + i - 1
                    # if start_loc < 0:
                    #     start_loc = 0

                    break
            end_loc = start_loc + self.n_samples + 1

            # Adjust trials per option
            # self.trials_per_option = np.zeros(self.num_options, dtype=int)

            logger.debug("Trial (start_loc, end_loc): (%s, %s)", start_loc, end_loc)
            # linear interpolation and add to numpy array
            for flash_index in flash_indices:
                for c in range(self.n_channels):
                    eeg_timestamps_adjusted = (
                        self.eeg_timestamps[start_loc:end_loc]
                        - self.eeg_timestamps[start_loc]
                    )

                    channel_data = np.interp(
                        self.trial_timestamps,
                        eeg_timestamps_adjusted,
                        self.eeg_data[start_loc:end_loc, c],
                    )

                    # add to raw ERP trials
                    self.erp_trials_raw[self.n_trials, c, 0 : self.n_samples] = (
                        channel_data
                    )
                    # self.decision_blocks_raw[self.decision_count, self.n_trials, c, 0:self.n_samples]

                    # if self.pp_type == "bandpass":
                    #     channel_data_2 = bandpass(channel_data[np.newaxis,:], self.pp_low, self.pp_high, self.pp_order, self.fsample)
                    #     channel_data = channel_data_2[0,:]

                    # # Add to the instance count
                    # self.trials_per_decision[flash_index] += 1

                    if self.plot_erp:
                        if flash_index == current_target:
                            self.axs1[c].plot(range(self.n_samples), channel_data)

                        elif (
                            self.non_target_plot == 99
                            or self.non_target_plot == flash_index
                        ):
                            self.axs2[c].plot(range(self.n_samples), channel_data)
                            self.non_target_plot = flash_index

                    # # add to processed ERP trials
                    # self.erp_trials[self.n_trials, c, 0:self.n_samples] = channel_data

                    # # Does the ensemble avearging
                    # self.decision_blocks[self.decision_count, flash_index, c, 0:self.n_samples] += channel_data

                # This is where to do preprocessing
                self.erp_trials_processed[
                    self.n_trials, : self.n_channels, : self.n_samples
                ] = self._preprocessing(
                    trial=self.erp_trials_raw[
                        self.n_trials, : self.n_channels, : self.n_samples
                    ],
                    option=self.pp_type,
                    order=self.pp_order,
                    fl=self.pp_low,
                    fh=self.pp_high,
                )

                # This is where to do artefact rejection
                self.erp_trials_processed[
                    self.n_trials, : self.n_channels, : self.n_samples
                ] = self._artefact_rejection(
                    trial=self.erp_trials_processed[
                        self.n_trials, : self.n_channels, : self.n_samples
                    ],
                    option=None,
                )

                # Add the raw trial to the raw decision blocks
                # self.decision_blocks_raw[self.decision_count, flash_index, 0:self.n_channels, 0:self.n_samples] +=  self.erp_trials_processed
                self.big_decision_blocks_raw[
                    self.decision_count,
                    flash_index,
                    int(self.trials_per_decision[flash_index] - 1),
                    0 : self.n_channels,
                    0 : self.n_samples,
                ] = self.erp_trials_raw[
                    self.n_trials, : self.n_channels, : self.n_samples
                ]

                self.big_decision_blocks_processed[
                    self.decision_count,
                    flash_index,
                    int(self.trials_per_decision[flash_index] - 1),
                    0 : self.n_channels,
                    0 : self.n_samples,
                ] = self.erp_trials_processed[
                    self.n_trials, : self.n_channels, : self.n_samples
                ]

                # self.trials_per_decision[flash_index] += 1
            # Reset for the next decision

            # iterate to next trial
            self.marker_count += 1
            self.n_trials += 1
            self.search_index = start_loc
            if self.online:
                time.sleep(0.000001)
