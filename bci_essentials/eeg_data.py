"""Module for managing BCI data.

This module provides data classes for different BCI paradigms.

It includes the loading of offline data in `xdf` format
or the live streaming of LSL data.

The loaded/streamed data is added to a buffer such that offline and
online processing pipelines are identical.

Data is pre-processed (using the `signal_processing` module), windowed,
and classified (using one of the `classification` sub-modules).

Classes
-------
- `EEG_data` : For processing continuous data in windows of a defined
length.

"""

import pyxdf
import time
import numpy as np

from pylsl import StreamInlet, resolve_byprop, StreamOutlet, StreamInfo
from pylsl.pylsl import IRREGULAR_RATE

from bci_essentials.signal_processing import notch, bandpass
from bci_essentials.classification.generic_classifier import Generic_classifier
from bci_essentials.classification.null_classifier import Null_classifier


# EEG data
class EEG_data:
    """Class that holds, windows, processes, and classifies EEG data.

    This class is used for the processing of continuous EEG data in
    windows of a defined length.

    It includes the loading of offline data in `xdf` format.

    """

    def __init__(self, classifier: Generic_classifier = Null_classifier()):
        """Initializes `EEG_data` class.

        Parameters
        ----------
        classifier : Generic_classifier
            The classifier used by EEG_data

        Attributes
        ----------
        explict_settings : bool
            Description of attribute `explict_settings`.
            - Initial value is `False`.
        stream_outlet : bool
            Description of attribute `stream_outlet`.
            - Initial value is `False`.
        ping_count : bool
            Description of attribute `ping_count`.
            - Initial value is `0`.
        ping_interval : bool
            Description of attribute `ping_interval`.
            - Initial value is `5`.
        resting_state_exists : bool
            Description of attribute `resting_state_exists`.
            - Initial value is `False`.

        """
        self._classifier = classifier

        self.explicit_settings = False
        self.stream_outlet = False
        self.ping_count = 0
        self.ping_interval = 5

        # resting state
        self.resting_state_exists = False

    # LOADING DATA
    # Explicit definition of settings, not recommended
    def edit_settings(
        self,
        user_id="0000",
        nchannels=8,
        channel_labels=["?", "?", "?", "?", "?", "?", "?", "?"],
        fsample=256,
        max_size=10000,
    ):
        """Explicit definition of settings.

        Change the settings for (...?)

        "not recommended."

        Parameters
        ----------
        user_id : str, *optional*
            The user ID.
            - Default is `"0000"`.
        nchannels : int, *optional*
            The number of channels.
            - Default is `8`.
        channel_labels : list of `str`, *optional*
            The channel labels.
            - Default is `["?", "?", "?", "?", "?", "?", "?", "?"]`.
        fsample : int, *optional*
            The sampling rate.
            - Default is `256`.
        max_size : int, *optional*
            Description of parameter `max_size`.
            - Default is `10000`.

        Returns
        -------
        `None`
            `self.explicit_settings` is set to `True`.

        """
        self.user_id = user_id  # user id
        self.nchannels = nchannels  # number of channels
        self.channel_labels = channel_labels  # EEG electrode placements
        self.fsample = fsample  # sampling rate
        self.max_size = max_size  # maximum size of eeg
        self.explicit_settings = (
            True  # settings are explicit and will not be updated based on headset data
        )

        if len(channel_labels) != self.nchannels:
            print("Channel locations do not fit number of channels!!!")
            self.channel_labels = ["?"] * self.nchannels

    # Load data from a variety of sources
    # Currently only suports .xdf format
    def load_offline_eeg_data(
        self, filename, format="xdf", subset=[], print_output=True
    ):
        """Loads offline data from a file.

        Currently only supports .xdf

        Parameters
        ----------
        filename : str
            The filename.
        format : str, *optional*
            The file format.
            - Default is `"xdf"`.
        subset : list of `int`, *optional*
            Description of parameter `subset`.
            - Default is `[]`.
        print_output : bool, *optional*
            Output is printed if `True`.
            - Default is `True`.

        Returns
        -------
        `None`
            `self` is updated.

        """
        self.subset = subset

        if format == "xdf":
            if print_output:
                print("loading ERP data from {}".format(filename))

            # load from xdf
            data, self.header = pyxdf.load_xdf(filename)

            # get the indexes of data
            for i in range(len(data)):
                namestring = data[i]["info"]["name"][0]
                typestring = data[i]["info"]["type"][0]
                if print_output:
                    print(namestring)
                    print(typestring)

                if typestring == "EEG":
                    self.eeg_index = i
                if "LSL_Marker_Strings" in typestring:
                    self.marker_index = i
                if "PythonResponse" in namestring:
                    self.response_index = i

            # fill up the marker and eeg buffers with the saved data
            try:
                self.marker_data = data[self.marker_index]["time_series"]
                self.marker_timestamps = data[self.marker_index]["time_stamps"]
            except Exception:
                print("Marker data not available")

            try:
                self.eeg_data = data[self.eeg_index]["time_series"]
                self.eeg_timestamps = data[self.eeg_index]["time_stamps"]
            except Exception:
                print("EEG data not available")

            try:
                self.response_data = data[self.response_index]["time_series"]
                self.response_timestamps = data[self.response_index]["time_stamps"]
            except Exception:
                print("Response data not available")

            # Unless explicit settings are desired, get settings from headset
            # if self.explicit_settings is False:
            self.__get_info_from_file(data, print_output)

        # support for other file types goes here

        # otherwise show an error
        else:
            print("Error: file format not supported")

    # Get metadata saved to the offline data file to fill in headset information
    def __get_info_from_file(self, data, print_output=True):
        """Get EEG metadata from the stream.

        Parameters
        ----------
        data : pylsl.StreamInlet - need to verify
            Data object from the LSL stream.
        print_output : bool, *optional*
            Output is printed if `True`.
            - Default is `True`.

        Returns:
        `None`
            `self` is updated.

        """

        self.headset_string = data[self.eeg_index]["info"]["name"][
            0
        ]  # headset name in string format
        self.fsample = float(
            data[self.eeg_index]["info"]["nominal_srate"][0]
        )  # sampling rate
        self.nchannels = int(
            data[self.eeg_index]["info"]["channel_count"][0]
        )  # number of channels

        # get chtypes, chunits
        self.ch_type = []
        self.ch_units = []
        for i in range(self.nchannels):
            # type
            ch_type = data[self.eeg_index]["info"]["desc"][0]["channels"][0]["channel"][
                i
            ]["type"][0]
            # send to lower case letters for mne
            ch_type = ch_type.lower()
            # save trigger channel as stim
            if ch_type == "trg":
                ch_type = "stim"
            # add to list
            self.ch_type.append(ch_type)

            # units
            ch_units = data[self.eeg_index]["info"]["desc"][0]["channels"][0][
                "channel"
            ][i]["unit"][0]
            self.ch_units.append(ch_units)

        self.channel_labels = []  # channel labels/locations, 'TRG' means trigger
        try:
            for i in range(self.nchannels):
                self.channel_labels.append(
                    data[self.eeg_index]["info"]["desc"][0]["channels"][0]["channel"][
                        i
                    ]["label"][0]
                )

        except Exception:
            for i in range(self.nchannels):
                self.channel_labels.append("?")

        if print_output:
            print(self.channel_labels)

        # if it is the DSI7 flex, relabel the channels, may want to make this more flexible in the future
        if print_output:
            print(self.headset_string)
        if self.headset_string == "DSI7":
            if print_output:
                print(self.channel_labels)
            # self.channel_labels[self.channel_labels.index('S1')] = 'O1'
            # self.channel_labels[self.channel_labels.index('S2')] = 'Pz'
            # self.channel_labels[self.channel_labels.index('S3')] = 'O2'

            self.nchannels = 7
            self.channel_labels.pop()

        if self.headset_string == "DSI24":
            self.nchannels = 23
            self.channel_labels.pop()

        # if self.headset_string == "EmotivDataStream-EEG":

        #     self.nchannels = 32
        #     self.channel_labels = self.channel_labels[3:-2]

        # if other headsets have quirks, they can be accomodated for here

        # If a subset is to be used, define a new nchannels, channel labels, and eeg data
        if self.subset != []:
            if print_output:
                print("A subset was defined")
                print("Original channels")
                print(self.channel_labels)

            self.nchannels = len(self.subset)
            self.subset_indices = []
            for s in self.subset:
                self.subset_indices.append(self.channel_labels.index(s))

            self.channel_labels = self.subset
            if print_output:
                print("Subset channels")
                print(self.channel_labels)

            # Apply the subset to the raw data
            self.eeg_data = self.eeg_data[:, self.subset_indices]

        else:
            self.subset_indices = list(range(0, self.nchannels))

        # send channel labels to classifier
        try:
            self._classifier.channel_labels = self.channel_labels
        except Exception:
            if print_output:
                print("no classifier defined")

        if print_output:
            print(self.headset_string)
            print(self.channel_labels)

    # ONLINE
    # stream data from an online source
    def stream_online_eeg_data(
        self,
        timeout=5,
        max_eeg_samples=1000000,
        max_marker_samples=100000,
        eeg_only=False,
        subset=[],
    ):
        """Stream data from an online source.

        Parameters
        ----------
        timeout : int, *optional*
            Description of parameter `timeout`.
            Default is `5`.
        max_eeg_samples : int, *optional*
            Description of parameter `max_eeg_samples`.
            - Default is `1000000`.
        max_marker_samples : int, *optional*
            Description of parameter `max_marker_samples`.
            - Default is `100000`.
        eeg_only : bool, *optional*
            Description of parameter `eeg_only`.
            - Default is `False`.
        subset : list of `int`, *optional*
            Description of parameter `subset`.
            - Default is `[]`.

        Returns
        -------
        `None`
            `self` is updated.

        """
        self.subset = subset

        print("printing incoming stream")

        # Unclear, but this is set twice...going to comment this out.
        # self.subset = subset

        wait_for_markers_flag = True
        wait_for_eeg_flag = True

        while wait_for_markers_flag is True or wait_for_eeg_flag is True:
            if eeg_only is True:
                wait_for_markers_flag = False

            if wait_for_markers_flag is True:
                # Try to resolve LSL marker stream
                try:
                    print("Resolving LSL marker stream... ")
                    marker_stream = resolve_byprop(
                        "type", "LSL_Marker_Strings", timeout=timeout
                    )
                    self.marker_inlet = StreamInlet(
                        marker_stream[0], processing_flags=0
                    )
                    print("Getting stream info...")
                    marker_info = self.marker_inlet.info()
                    print("The marker stream's XML meta-data is: ")
                    print(marker_info.as_xml())

                    wait_for_markers_flag = False

                except Exception:
                    print("No marker stream currently available")
                    wait_for_markers_flag = True

            if wait_for_eeg_flag is True:
                # Try to resolve EEG marker stream
                try:
                    print("Resolving LSL EEG stream... ")
                    eeg_stream = resolve_byprop("type", "EEG", timeout=timeout)
                    # We may want to explore better the properties of the stream inlet call. For refrence see here: https://github.com/labstreaminglayer/pylsl/blob/master/pylsl/pylsl.py
                    self.eeg_inlet = StreamInlet(eeg_stream[0], processing_flags=0)
                    print("Getting stream info...")
                    eeg_info = self.eeg_inlet.info()
                    print("The EEG stream's XML meta-data is: ")
                    print(eeg_info.as_xml())
                    print(eeg_info)
                    # print(eeg_info.created_at)

                    # if there are no explicit settings
                    if self.explicit_settings is False:
                        self.__get_info_from_stream()

                    wait_for_eeg_flag = False

                except Exception as e:
                    print("No EEG stream currently available")
                    print(e)  # print the exception
                    wait_for_eeg_flag = True

            # Exit if one or both streams are unavailable
            if wait_for_markers_flag is True or wait_for_eeg_flag is True:
                if wait_for_markers_flag is True:
                    print("Waiting for marker stream")
                if wait_for_eeg_flag is True:
                    print("Waiting for EEG stream")

                print("Waiting 5 seconds for streams to become available...")
                print("Press Ctrl-C to exit")
                time.sleep(5)

        self.marker_data = []
        self.marker_timestamps = []
        self.eeg_data = []
        self.eeg_timestamps = []

        self.marker_data = np.array(self.marker_data)
        self.marker_timestamps = np.array(self.marker_timestamps)
        self.eeg_data = np.array(self.eeg_data)
        self.eeg_timestamps = np.array(self.eeg_timestamps)

    # Get headset data from stream
    def __get_info_from_stream(self):
        """Get headset data from stream.

        Returns
        -------
        `None`
            `self` is updated.
        """
        # get info obect from stream
        eeg_info = self.eeg_inlet.info()

        self.headset_string = eeg_info.name()  # headset name in string format
        self.fsample = float(eeg_info.nominal_srate())  # sampling rate
        self.nchannels = int(eeg_info.channel_count())  # number of channels

        # get online channel types and units

        # iterate through children of <"channels"> to get the channel labels
        ch = eeg_info.desc().child("channels").child("channel")
        self.channel_labels = []  # channel labels/locations, 'TRG' means trigger
        print("num channels = ", self.nchannels)
        for i in range(self.nchannels):
            name = ch.child_value("name")
            if name == "":
                name = ch.child_value("label")
            self.channel_labels.append(name)
            # go to next sibling
            ch = ch.next_sibling()

        # if it is the DSI7 flex, relabel the channels, may want to make this more flexible in the future
        if self.headset_string == "DSI7":
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # print(self.channel_labels)
            # self.channel_labels[self.channel_labels.index('S1')] = 'O1'
            # self.channel_labels[self.channel_labels.index('S2')] = 'Pz'
            # self.channel_labels[self.channel_labels.index('S3')] = 'O2'

            self.channel_labels.pop()
            self.nchannels = 7

        if self.headset_string == "DSI24":
            self.channel_labels.pop()
            self.nchannels = 23

        # if self.headset_string == "EmotivDataStream-EEG":

        #     self.nchannels = 32
        #     self.channel_labels = self.channel_labels[3:-2] #Accounting for all the extra parts in EmotivFlex

        # if other headsets have quirks, they can be accomodated for here

        # If a subset is to be used, define a new nchannels, channel labels, and eeg data
        if self.subset != []:
            print("A subset was defined")
            print("Original channels")
            print(self.channel_labels)

            self.nchannels = len(self.subset)
            self.subset_indices = []
            for s in self.subset:
                self.subset_indices.append(self.channel_labels.index(s))

            self.channel_labels = self.subset
            print("Subset channels")
            print(self.channel_labels)

        else:
            print("No subset was defined. Using all channels")
            self.subset_indices = list(range(0, self.nchannels))

        # send channel labels to classifier
        try:
            self._classifier.channel_labels = self.channel_labels
        except Exception:
            print("no classifier defined")

        # Print some headset info
        print(self.headset_string)
        print(self.channel_labels)

    # Get new data from stream
    def _pull_data_from_stream(
        self, include_markers=True, include_eeg=True, return_eeg=False
    ):
        """Get new data from stream.

        Parameters
        ----------
        include_markers : bool, *optional*
            Whether to include marker data in the pull.
            - Default is `True`.
        include_eeg : bool, *optional*
            Whether to include EEG data in the pull.
            - Default is `True`.
        return_eeg : bool, *optional*
            Whether to return EEG data.
            - Default is `False`.

        Returns
        -------
        new_eeg_timestamps : list of `float`
            Timestamps of new EEG data.
            Only returns if `return_eeg` is `True`.
        new_eeg_data : pylsl.StreamInlet - need to verify
            Data object from the LSL stream.
            Only returns if `return_eeg` is `True`.

        """
        # Pull chunks of new data
        if include_markers:
            # pull marker chunk
            new_marker_data, new_marker_timestamps = self.marker_inlet.pull_chunk(
                timeout=0.1
            )

            # pull marker time correction
            self.marker_time_correction = self.marker_inlet.time_correction()

            # apply time correction
            new_marker_timestamps = [
                new_marker_timestamps[i] + self.marker_time_correction
                for i in range(len(new_marker_timestamps))
            ]

            # save the marker data to the data object
            self.marker_data = np.array(list(self.marker_data) + new_marker_data)
            self.marker_timestamps = np.array(
                list(self.marker_timestamps) + new_marker_timestamps
            )

        if include_eeg:
            # Add a try exception around pulling the chunks
            new_eeg_data, new_eeg_timestamps = self.eeg_inlet.pull_chunk(timeout=0.1)

            # Interrogate the pulled chunk for quality to make sure that it's size makes sense and is not null.
            # This should be done later with a python data structure.
            # Check to see if any nan vals are present
            nanValsPresent = np.isnan(np.dot(new_eeg_data, new_eeg_data))
            if nanValsPresent:
                raise ValueError(
                    "Seems that there are null values in the EEG chunk pulled. Returning"
                )

            # Convert into a numpy array
            new_eeg_data = np.array(new_eeg_data)

            # Handle the case when you are using subsets - seemingly always have a subset.
            if self.subset != []:
                # add try/except phrase around pulling the subset in the chance that something is wrong.
                # The bug is around the array size being 1-dimensional but us accessing 2 dimensions.
                new_eeg_data = new_eeg_data[:, self.subset_indices]

            # if time is in milliseconds, divide by 1000, works for sampling rates above 10Hz
            try:
                if self.time_units == "milliseconds":
                    new_eeg_timestamps = [
                        (new_eeg_timestamps[i] / 1000)
                        for i in range(len(new_eeg_timestamps))
                    ]

            # If time units are not defined then define them
            except Exception:
                dif_low = -2
                dif_high = -1
                while new_eeg_timestamps[dif_high] - new_eeg_timestamps[dif_low] == 0:
                    dif_low -= 1
                    dif_high -= 1

                if new_eeg_timestamps[dif_high] - new_eeg_timestamps[dif_low] > 0.1:
                    new_eeg_timestamps = [
                        (new_eeg_timestamps[i] / 1000)
                        for i in range(len(new_eeg_timestamps))
                    ]
                    self.time_units = "milliseconds"
                else:
                    self.time_units = "seconds"

            # apply time correction, this is essential for headsets like neurosity which have their own clock
            self.eeg_time_correction = self.eeg_inlet.time_correction()

            # MAYBE DONT NEED THIS WITH NEW PROC SETTINGS
            new_eeg_timestamps = [
                new_eeg_timestamps[i] + self.eeg_time_correction
                for i in range(len(new_eeg_timestamps))
            ]

            # save the EEG data to the data object
            try:
                self.eeg_data = np.concatenate((self.eeg_data, new_eeg_data))
            except Exception:
                self.eeg_data = new_eeg_data

            # save the marker data to the data object
            self.eeg_timestamps = np.array(
                list(self.eeg_timestamps) + new_eeg_timestamps
            )

        # If the outlet exists send a ping
        if self.stream_outlet:
            self.ping_count += 1
            if self.ping_count % self.ping_interval:
                self.outlet.push_sample(["ping"])

        # Return eeg
        if return_eeg:
            return new_eeg_timestamps, new_eeg_data

    def save_data(self, directory_name):
        """Save the data from different stages.

        Creates a directory with x files. Includes raw EEG, markers,
        processed EEG, features.

        **NOT IMPLEMENTED YET**

        Parameters
        ----------
        directory_name : str
            Name of the directory to save the data to.

        Returns
        -------
        data_pickle : pickle
            Saves the data as a pickle file. Includes raw EEG, markers,
            processed EEG, features.

        """

    def mne_export_as_raw(self):
        """MNE export EEG as RawArray.

        Exports the EEG data as a MNE RawArray object.

        **Requires MNE**

        Returns
        -------
        raw_array : mne.io.RawArray
            MNE RawArray object.

        """
        print("mne_export_as_raw has not been implemented yet")
        # Check for mne
        try:
            import mne
        except Exception:
            print("Could not import mne, you may have to install (pip install mne)")

        # create info from metadata
        info = mne.create_info(
            ch_names=self.channel_labels, sfreq=self.fsample, ch_types="eeg"
        )

        # create the MNE epochs, pass in the raw

        # make sure that units match
        raw_data = self.eeg_data.transpose()
        raw_array = mne.io.RawArray(data=raw_data, info=info)

        # change the last column of epochs array events to be the class labels
        # raw_array.events[:, -1] = self.labels

        return raw_array

    def mne_export_as_epochs(self):
        """MNE export EEG as EpochsArray.

        Exports the EEG data as a MNE EpochsArray object.

        **Requires MNE**

        Returns
        -------
        epochs_array : mne.EpochsArray
            MNE EpochsArray object.

        """
        # Check for mne
        try:
            import mne
        except Exception:
            print("Could not import mne, you may have to install (pip install mne)")

        # create info from metadata
        info = mne.create_info(
            ch_names=self.channel_labels, sfreq=self.fsample, ch_types=self.ch_type
        )

        # create the MNE epochs, pass in the raw

        # make sure that units match
        epoch_data = self.raw_eeg_windows.copy()
        for i, u in enumerate(self.ch_units):
            if u == "microvolts":
                # convert to volts
                epoch_data[:, i, :] = epoch_data[:, i, :] / 1000000

        epochs_array = mne.EpochsArray(data=epoch_data, info=info)

        # change the last column of epochs array events to be the class labels
        epochs_array.events[:, -1] = self.labels

        return epochs_array

    def mne_export_resting_state_as_raw(self):
        """MNE export resting state EEG as RawArray.

        Exports the resting state EEG data as a MNE RawArray object.

        **Requires MNE**

        Returns
        -------
        raw_array : mne.io.RawArray
            MNE RawArray object.

        """
        print("mne_export_as_raw has not been implemented yet")
        # Check for mne
        try:
            import mne
        except Exception:
            print("Could not import mne, you may have to install (pip install mne)")

        # create info from metadata
        info = mne.create_info(
            ch_names=self.channel_labels, sfreq=self.fsample, ch_types="eeg"
        )

        try:
            # create the MNE epochs, pass in the raw

            # make sure that units match
            raw_data = self.rest_windows[0, :, :]
            raw_array = mne.io.RawArray(data=raw_data, info=info)

            # change the last column of epochs array events to be the class labels
            # raw_array.events[:, -1] = self.labels

        except Exception:
            # could not find resting state data, sending the whole collection instead
            print(
                "NO PROPER RESTING STATE DATA FOUND, SENDING ALL OF THE EEG DATA INSTEAD"
            )
            raw_data = self.eeg_data.transpose()
            raw_array = mne.io.RawArray(data=raw_data, info=info)

        return raw_array

    # SIGNAL PROCESSING
    # Preprocessing goes here (windows are nchannels by nsamples)
    def _preprocessing(self, window, option=None, order=5, fc=60, fl=10, fh=50):
        """Signal preprocessing.

        Preprocesses the signal using one of the methods from the
        `signal_processing.py` module.

        Parameters
        ----------
        window : numpy.ndarray
            Window of EEG data.
            2D array containing data with `float` type.

            shape = (`N_channels`,`N_samples`)
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
        new_window : numpy.ndarray
            Preprocessed window of EEG data.
            2D array containing data with `float` type.

            shape = (`N_channels`,`N_samples`)

        """
        # do nothing
        if option is None:
            new_window = window
            return new_window

        if option == "notch":
            new_window = notch(window, fc=60, Q=30, fsample=self.fsample)
            return new_window

        if option == "bandpass":
            new_window = bandpass(window, fl, fh, order, self.fsample)
            return new_window

        # other preprocessing options go here

    # Artefact rejection goes here (windows are nchannels by nsamples)
    def _artefact_rejection(self, window, option=None):
        """Artefact rejection.

        Parameters
        ----------
        window : numpy.ndarray
            Window of EEG data.
            2D array containing data with `float` type.

            shape = (`N_channels`,`N_samples`)
        option : str, *optional*
            Artefact rejection option. Options include:
            - Nothing has been implemented yet.
            - Default is `None`.

        Returns
        -------
        new_window : numpy.ndarray
            Artefact rejected window of EEG data.
            2D array containing data with `float` type.

            shape = (`N_channels`,`N_samples`)

        """
        # do nothing
        if option is None:
            new_window = window
            return new_window

        # other preprocessing options go here\

    # I don't think this is being used....
    def __package_resting_state_data(self):
        """Package resting state data.

        Returns
        -------
        `None`
            `self.rest_windows` is updated.

        """
        try:
            print("Packaging resting state data")

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
                # Increment the EEG until just past the marker timestamp
                while current_time < self.marker_timestamps[i]:
                    current_timestamp_loc += 1
                    current_time = self.eeg_timestamps[current_timestamp_loc]

                # get eyes open start times
                if self.marker_data[i][0] == "Start Eyes Open RS: 1":
                    eyes_open_start_time.append(self.marker_timestamps[i])
                    eyes_open_start_loc.append(current_timestamp_loc - 1)
                    # print("received eyes open start")

                # get eyes open end times
                if self.marker_data[i][0] == "End Eyes Open RS: 1":
                    eyes_open_end_time.append(self.marker_timestamps[i])
                    eyes_open_end_loc.append(current_timestamp_loc)
                    # print("received eyes open end")

                # get eyes closed start times
                if self.marker_data[i][0] == "Start Eyes Closed RS: 2":
                    eyes_closed_start_time.append(self.marker_timestamps[i])
                    eyes_closed_start_loc.append(current_timestamp_loc - 1)
                    # print("received eyes closed start")

                # get eyes closed end times
                if self.marker_data[i][0] == "End Eyes Closed RS: 2":
                    eyes_closed_end_time.append(self.marker_timestamps[i])
                    eyes_closed_end_loc.append(current_timestamp_loc)
                    # print("received eyes closed end")

                # get rest start times
                if self.marker_data[i][0] == "Start Rest for RS: 0":
                    rest_start_time.append(self.marker_timestamps[i])
                    rest_start_loc.append(current_timestamp_loc - 1)
                    # print("received rest start")
                # get rest end times
                if self.marker_data[i][0] == "End Rest for RS: 0":
                    rest_end_time.append(self.marker_timestamps[i])
                    rest_end_loc.append(current_timestamp_loc)
                    # print("received rest end")

            # Eyes open
            # Get duration, nsmaples

            if len(eyes_open_end_loc) > 0:
                duration = np.floor(eyes_open_end_time[0] - eyes_open_start_time[0])
                nsamples = int(duration * self.fsample)

                self.eyes_open_timestamps = np.array(range(nsamples)) / self.fsample
                self.eyes_open_windows = np.ndarray(
                    (len(eyes_open_start_time), self.nchannels, nsamples)
                )
                # Now copy EEG for these windows
                for i in range(len(eyes_open_start_time)):
                    # For each channel of the EEG, interpolate to uniform sampling rate
                    for c in range(self.nchannels):
                        # First, adjust the EEG timestamps to start from zero
                        eeg_timestamps_adjusted = (
                            self.eeg_timestamps[
                                eyes_open_start_loc[i] : eyes_open_end_loc[i]
                            ]
                            - self.eeg_timestamps[eyes_open_start_loc[i]]
                        )

                        # Second, interpolate to timestamps at a uniform sampling rate
                        channel_data = np.interp(
                            self.eyes_open_timestamps,
                            eeg_timestamps_adjusted,
                            self.eeg_data[
                                eyes_open_start_loc[i] : eyes_open_end_loc[i], c
                            ],
                        )

                        # Third, add to the EEG window
                        self.eyes_open_windows[i, c, :] = channel_data
                        self.eyes_open_timestamps

            print("Done packaging resting state data")

            # Eyes closed

            if len(eyes_closed_end_loc) > 0:
                # Get duration, nsmaples
                duration = np.floor(eyes_closed_end_time[0] - eyes_closed_start_time[0])
                nsamples = int(duration * self.fsample)

                self.eyes_closed_timestamps = np.array(range(nsamples)) / self.fsample
                self.eyes_closed_windows = np.ndarray(
                    (len(eyes_closed_start_time), self.nchannels, nsamples)
                )
                # Now copy EEG for these windows
                for i in range(len(eyes_closed_start_time)):
                    # For each channel of the EEG, interpolate to uniform sampling rate
                    for c in range(self.nchannels):
                        # First, adjust the EEG timestamps to start from zero
                        eeg_timestamps_adjusted = (
                            self.eeg_timestamps[
                                eyes_closed_start_loc[i] : eyes_closed_end_loc[i]
                            ]
                            - self.eeg_timestamps[eyes_closed_start_loc[i]]
                        )

                        # Second, interpolate to timestamps at a uniform sampling rate
                        channel_data = np.interp(
                            self.eyes_closed_timestamps,
                            eeg_timestamps_adjusted,
                            self.eeg_data[
                                eyes_closed_start_loc[i] : eyes_closed_end_loc[i], c
                            ],
                        )

                        # Third, add to the EEG window
                        self.eyes_closed_windows[i, c, :] = channel_data
                        self.eyes_closed_timestamps

            # Rest
            if len(rest_end_loc) > 0:
                # Get duration, nsmaples
                while rest_end_time[0] < rest_start_time[0]:
                    rest_end_time.pop(0)
                    rest_end_loc.pop(0)

                duration = np.floor(rest_end_time[0] - rest_start_time[0])

                nsamples = int(duration * self.fsample)

                self.rest_timestamps = np.array(range(nsamples)) / self.fsample
                self.rest_windows = np.ndarray(
                    (len(rest_start_time), self.nchannels, nsamples)
                )
                # Now copy EEG for these windows
                for i in range(len(rest_start_time)):
                    # For each channel of the EEG, interpolate to uniform sampling rate
                    for c in range(self.nchannels):
                        # First, adjust the EEG timestamps to start from zero
                        eeg_timestamps_adjusted = (
                            self.eeg_timestamps[rest_start_loc[i] : rest_end_loc[i]]
                            - self.eeg_timestamps[rest_start_loc[i]]
                        )

                        # Second, interpolate to timestamps at a uniform sampling rate
                        channel_data = np.interp(
                            self.rest_timestamps,
                            eeg_timestamps_adjusted,
                            self.eeg_data[rest_start_loc[i] : rest_end_loc[i], c],
                        )

                        # Third, add to the EEG window
                        self.rest_windows[i, c, :] = channel_data
                        self.rest_timestamps
        except Exception:
            print("Failed to package resting state data")

    # main
    # add pp_low, pp_high, pp_order, subset

    def main(
        self,
        buffer=0.01,
        eeg_start=0,
        max_channels=64,
        max_samples=2560,
        max_windows=1000,
        max_loops=1000000,
        training=True,
        online=True,
        train_complete=False,
        iterative_training=False,
        live_update=False,
        print_markers=True,
        print_training=True,
        print_fit=True,
        print_performance=True,
        print_predict=True,
        pp_type="bandpass",  # preprocessing method
        pp_low=1,  # bandpass lower cutoff
        pp_high=40,  # bandpass upper cutoff
        pp_order=5,  # bandpass order
    ):
        """Main function of `EEG_data` class.

        Runs a while loop that reads in EEG data from the `EEG_data` object
        and processes it. Can be used in `online` or `offline` mode.
        - If in `online` mode, then the loop will continuously try to read
        in data from the `EEG_data` object and process it. The loop will
        terminate when `max_loops` is reached, or when manually terminated.
        - If in `offline` mode, then the loop will read in all of the data
        at once, process it, and then terminate.

        Parameters
        ----------
        buffer : float, *optional*
            Buffer time for EEG sampling in `online` mode (seconds).
            - Default is `0.01`.
        eeg_start : int, *optional*
            Start time for EEG sampling (seconds).
            - Default is `0`.
        max_channels : int, *optional*
            Maximum number of EEG channels to read in.
            - Default is `64`.
        max_samples : int, *optional*
            Maximum number of EEG samples to read in per window.
            - Default is `2560`.
        max_windows : int, *optional*
            Maximum number of windows to read in per loop (?).
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
            live updates on window classification.
            - Default is `False`.
        print_markers : bool, *optional*
            Flag to indicate if the markers will be printed to the console.
            - Default is `True`.
        print_training : bool, *optional*
            Flag to indicate if the training progress will be printed to the
            console.
            - Default is `True`.
        print_fit : bool, *optional*
            Flag to indicate if the classifier fit will be printed to the
            console.
            - Default is `True`.
        print_performance : bool, *optional*
            Flag to indicate if the classifier performance will be printed
            to the console.
            - Default is `True`.
        print_predict : bool, *optional*
            Flag to indicate if the classifier predictions will be printed
            to the console.
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

        Returns
        -------
        `None`

        """

        # if this is the first time this function is being called for a given dataset then run some initialization
        if eeg_start == 0:
            self.window_end_buffer = buffer
            search_index = 0

            # initialize windows and labels
            current_raw_eeg_windows = np.zeros((max_windows, max_channels, max_samples))
            current_processed_eeg_windows = current_raw_eeg_windows
            current_labels = np.zeros((max_windows))

            self.raw_eeg_windows = np.zeros((max_windows, max_channels, max_samples))
            self.processed_eeg_windows = self.raw_eeg_windows
            self.labels = np.zeros((max_windows))  # temporary labels
            self.training_labels = np.zeros((max_windows))  # permanent training labels

            # initialize the numbers of markers and windows to zero
            self.marker_count = 0
            current_nwindows = 0
            self.nwindows = 0

            #
            self.num_online_selections = 0
            self.online_selection_indices = []
            self.online_selections = []

            # initialize loop count
            loops = 0

        # start the main loop, stops after pulling now data, max_loops times
        while loops < max_loops:
            #
            if loops % 100 == 0:
                if print_markers:
                    print(loops)

            if loops == max_loops - 1:
                print("last loop")

            # if offline, then all data is already loaded, no need to iterate
            if online is False:
                loops = max_loops

            # if online, then pull new data with each iteration
            if online:
                # Adding a try catch here, incase error happens in the _pull_data_from_stream() call.
                try:
                    self._pull_data_from_stream()
                except Exception as e:
                    print(f"An error has occurred when pulling data: {e}")
                    continue

                # Create a stream to send markers back to Unity, but only create the stream once
                if self.stream_outlet is False:
                    # define the stream information
                    info = StreamInfo(
                        name="PythonResponse",
                        type="BCI",
                        channel_count=1,
                        nominal_srate=IRREGULAR_RATE,
                        channel_format="string",
                        source_id="pyp30042",
                    )
                    # print(info)
                    # create the outlet
                    self.outlet = StreamOutlet(info)

                    # next make an outlet
                    print("the outlet exists")
                    self.stream_outlet = True

                    # Push the data
                    self.outlet.push_sample(["This is the python response stream"])

            # check if there is an available marker, if not, break and wait for more data
            while len(self.marker_timestamps) > self.marker_count:
                loops = 0

                # If the marker contains a single string, not including ',' and begining with a alpha character, then it is an event message
                if (
                    len(self.marker_data[self.marker_count][0].split(",")) == 1
                    and self.marker_data[self.marker_count][0][0].isalpha()
                ):
                    # send feedback to unity if there is an available outlet
                    if self.stream_outlet:
                        # send feedback for each marker that you receive
                        self.outlet.push_sample(
                            [
                                "marker received : {}".format(
                                    self.marker_data[self.marker_count][0]
                                )
                            ]
                        )

                    ############
                    if print_markers:
                        print(self.marker_data[self.marker_count][0])

                    # once all resting state data is collected then go and compile it
                    if (
                        self.marker_data[self.marker_count][0]
                        == "Done with all RS collection"
                    ):
                        self.package_resting_state_data()
                        self.marker_count += 1

                    elif self.marker_data[self.marker_count][0] == "Trial Started":
                        if print_markers:
                            print("Trial started")
                        # Note that a marker occured, but do nothing else
                        self.marker_count += 1

                    elif self.marker_data[self.marker_count][0] == "Trial Ends":
                        if print_markers:
                            print("Trial ended")

                        # Trim the unused ends of numpy arrays
                        current_raw_eeg_windows = current_raw_eeg_windows[
                            0:current_nwindows, 0 : self.nchannels, 0 : self.nsamples
                        ]
                        current_processed_eeg_windows = current_processed_eeg_windows[
                            0:current_nwindows, 0 : self.nchannels, 0 : self.nsamples
                        ]
                        current_labels = current_labels[0:current_nwindows]

                        # TRAIN
                        if training:
                            self._classifier.add_to_train(
                                current_processed_eeg_windows,
                                current_labels,
                                print_training=print_training,
                            )

                            if print_training:
                                print(
                                    current_nwindows,
                                    " windows and labels added to training set",
                                )

                            # if iterative training is on and active then also make a prediction
                            if iterative_training:
                                if print_predict:
                                    print(
                                        "Added current samples to training set, now making a prediction"
                                    )
                                prediction = self._classifier.predict(
                                    current_processed_eeg_windows,
                                    print_predict=print_predict,
                                )

                                # Send the prediction to Unity
                                if print_predict:
                                    print(
                                        "{} was selected by the iterative classifier, sending to Unity".format(
                                            prediction
                                        )
                                    )
                                # pick a sample to send an wait for a bit

                                # if online, send the packet to Unity
                                if online:
                                    self.outlet.push_sample(["{}".format(prediction)])

                        # PREDICT
                        elif train_complete and current_nwindows != 0:
                            if print_predict:
                                print(
                                    "making a prediction based on ",
                                    current_nwindows,
                                    " windows",
                                )

                            if current_nwindows == 0:
                                print("No windows to make a decision")
                                self.marker_count += 1
                                break

                            # save the online selection indices
                            selection_inds = list(
                                range(self.nwindows - current_nwindows, self.nwindows)
                            )
                            self.online_selection_indices.append(selection_inds)

                            # make the prediciton
                            try:
                                prediction = self._classifier.predict(
                                    current_processed_eeg_windows, print_predict
                                )
                                self.online_selections.append(prediction)

                                if print_predict:
                                    print("Recieved prediction from classifier")

                                    # Send the prediction to Unity
                                    print(
                                        "{} was selected, sending to Unity".format(
                                            prediction
                                        )
                                    )

                                # if online, send the packet to Unity
                                if online:
                                    self.outlet.push_sample(["{}".format(prediction)])

                            except Exception:
                                if print_predict:
                                    print("This classification failed...")

                        # OH DEAR
                        else:
                            print("Unable to classify... womp womp")

                        # Reset windows and labels
                        self.marker_count += 1
                        current_nwindows = 0
                        current_raw_eeg_windows = np.zeros(
                            (max_windows, max_channels, max_samples)
                        )
                        current_processed_eeg_windows = current_raw_eeg_windows
                        current_labels = np.zeros((max_windows))

                    # If training completed then train the classifier
                    # This is confusing.
                    elif (
                        self.marker_data[self.marker_count][0] == "Training Complete"
                        and train_complete is False
                    ):
                        if print_training:
                            print("Training the classifier")

                        self._classifier.fit(
                            print_fit=print_fit, print_performance=print_performance
                        )
                        train_complete = True
                        training = False
                        self.marker_count += 1

                    elif self.marker_data[self.marker_count][0] == "Update Classifier":
                        if print_training:
                            print("Retraining the classifier")

                        self._classifier.fit(
                            print_fit=print_fit, print_performance=print_performance
                        )

                        iterative_training = True
                        if online:
                            live_update = True

                        self.marker_count += 1

                    else:
                        self.marker_count += 1

                    if online:
                        time.sleep(0.01)
                    loops += 1
                    continue

                # Get marker info
                marker_info = self.marker_data[self.marker_count][0].split(",")

                self.paradigm_string = marker_info[0]
                self.num_options = int(marker_info[1])
                label = int(marker_info[2])
                self.window_length = float(marker_info[3])  # window length
                if (
                    len(marker_info) > 4
                ):  # if longer, collect this info and maybe it can be used by the classifier
                    self.meta = []
                    for i in range(4, len(marker_info)):
                        self.meta.__add__([marker_info[i]])

                    # Load the correct SSVEP freqs
                    if marker_info[0] == "ssvep":
                        self._classifier.target_freqs = [1] * (len(marker_info) - 4)
                        self._classifier.sampling_freq = self.fsample
                        for i in range(4, len(marker_info)):
                            self._classifier.target_freqs[i - 4] = float(marker_info[i])
                            # print("changed ", i-4, "target frequency to", marker_info[i])

                # Check if the whole EEG window corresponding to the marker is available
                end_time_plus_buffer = (
                    self.marker_timestamps[self.marker_count]
                    + self.window_length
                    + buffer
                )

                # If we don't have the full window then pull more data, only do this online
                if self.eeg_timestamps[-1] <= end_time_plus_buffer:
                    if online:
                        break
                    if online is False:
                        self.marker_count += 1
                        break

                if print_markers:
                    print(marker_info)

                # send feedback to unity if there is an available outlet
                if self.stream_outlet:
                    print("sending feedback to Unity")
                    # send feedback for each marker that you receive
                    self.outlet.push_sample(
                        [
                            "marker received : {}".format(
                                self.marker_data[self.marker_count][0]
                            )
                        ]
                    )

                # Find the start time for the window based on the marker timestamp
                start_time = self.marker_timestamps[self.marker_count]

                # Find the number of samples per window
                self.nsamples = int(self.window_length * self.fsample)

                # set the window timestamps at exactly the sampling frequency
                self.window_timestamps = np.arange(self.nsamples) / self.fsample

                # locate the indices of the window in the eeg data
                for i, s in enumerate(self.eeg_timestamps[search_index:-1]):
                    if s > start_time:
                        start_loc = search_index + i - 1
                        break

                # Get the end location for the window
                end_loc = int(start_loc + self.nsamples + 1)

                # For each channel of the EEG, interpolate to uniform sampling rate
                for c in range(self.nchannels):
                    # First, adjust the EEG timestamps to start from zero
                    eeg_timestamps_adjusted = (
                        self.eeg_timestamps[start_loc:end_loc]
                        - self.eeg_timestamps[start_loc]
                    )

                    # Second, interpolate to timestamps at a uniform sampling rate
                    channel_data = np.interp(
                        self.window_timestamps,
                        eeg_timestamps_adjusted,
                        self.eeg_data[start_loc:end_loc, c],
                    )

                    # Third, sdd to the EEG window
                    current_raw_eeg_windows[
                        current_nwindows, c, 0 : self.nsamples
                    ] = channel_data

                # This is where to do preprocessing
                current_processed_eeg_windows[
                    current_nwindows, : self.nchannels, : self.nsamples
                ] = self._preprocessing(
                    window=current_raw_eeg_windows[
                        current_nwindows, : self.nchannels, : self.nsamples
                    ],
                    option=pp_type,
                    order=pp_order,
                    fl=pp_low,
                    fh=pp_high,
                )

                # This is where to do artefact rejection
                current_processed_eeg_windows[
                    current_nwindows, : self.nchannels, : self.nsamples
                ] = self._artefact_rejection(
                    window=current_processed_eeg_windows[
                        current_nwindows, : self.nchannels, : self.nsamples
                    ],
                    option=None,
                )

                # Add the label if it exists, otherwise set a flag of -1 to denote that there is no label
                # if training:
                #     current_labels[current_nwindows] = label
                # else:
                #     current_labels[current_nwindows] = -1
                current_labels[current_nwindows] = label

                # copy to the eeg_data object
                self.raw_eeg_windows[
                    self.nwindows, 0 : self.nchannels, 0 : self.nsamples
                ] = current_raw_eeg_windows[
                    current_nwindows, 0 : self.nchannels, 0 : self.nsamples
                ]
                self.processed_eeg_windows[
                    self.nwindows, 0 : self.nchannels, 0 : self.nsamples
                ] = current_processed_eeg_windows[
                    current_nwindows, 0 : self.nchannels, 0 : self.nsamples
                ]
                self.labels[self.nwindows] = current_labels[current_nwindows]

                # Send live updates
                if live_update:
                    try:
                        if self.nsamples != 0:
                            pred = self._classifier.predict(
                                current_processed_eeg_windows[
                                    current_nwindows,
                                    0 : self.nchannels,
                                    0 : self.nsamples,
                                ],
                                print_predict=print_predict,
                            )
                            self.outlet.push_sample(["{}".format(int(pred[0]))])
                    except Exception:
                        print("unable to classify this window")

                # iterate to next window
                self.marker_count += 1
                current_nwindows += 1
                self.nwindows += 1
                search_index = start_loc

            # Wait a short period of time and then try to pull more data
            if online:
                time.sleep(0.00001)
            loops += 1

        # Trim all the data
        self.raw_eeg_windows = self.raw_eeg_windows[
            0 : self.nwindows, 0 : self.nchannels, 0 : self.nsamples
        ]
        self.processed_eeg_windows = self.processed_eeg_windows[
            0 : self.nwindows, 0 : self.nchannels, 0 : self.nsamples
        ]
        self.labels = self.labels[0 : self.nwindows]
