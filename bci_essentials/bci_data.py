# Data classes for BCI
# Written by Brian Irvine on 08/06/2021
# Updated by Brian Irvine on 12/01/2021

# This library contains mdoules for the processing of EEG data for BCI applications, supported applications
# are currently P300 ERP and SSVEP, modules are including for running online and offline in an identical fashion

# APPLICATIONS
# 1. Load  offline data
# 2. Stream online data
# 3. Provide options for visualizing BCI data


# LIMITATIONS
# 1. currently will not handle ERP sessions longer that 10000 markers in duration to avoid latency cause my dynamic sizing of numpy ndarrays,
#   this number can be increased by changing the max_windows variable in the ERP_data class

"""
Module for managing BCI data

This module provides data classes for different BCI paradigms. It includes the loading of offline data in xdf format or
the live streaming of LSL data. The loaded/streamed data is added to a buffer such that offline and online processing
pipelines are identical. Data is pre-processed (using the signal_processing module), windowed, and classified (using
the classification module).

Classes:
EEG_data - for processing continuous data in windows of a defined length
ERP_data - for processing P300 or other Event Related Potentials (ERP)
"""

import sys
import pyxdf
import time
import numpy as np
import matplotlib.pyplot as plt

import scipy

from pylsl import StreamInlet, resolve_byprop, StreamOutlet, StreamInfo, proc_dejitter
from pylsl.pylsl import IRREGULAR_RATE

#from bci_essentials.bci_data_settings import *
from bci_essentials.visuals import *
from bci_essentials.signal_processing import *
from bci_essentials.classification import *

# EEG data
class EEG_data():
    """
    Class that holds, windows, processes, and classifies EEG data
    """
    def __init__(self):
        """
        Howdy
        """
        self.explicit_settings = False
        self.classifier_defined = False
        self.stream_outlet = False
        self.ping_count = 0
        self.ping_interval = 5

        # resting state
        self.resting_state_exists = False

    # LOADING DATA
    # Explicit definition of settings, not recommended 
    def edit_settings(self, user_id='0000', nchannels=8, channel_labels=['?','?','?','?','?','?','?','?'], fsample=256, max_size=10000):
        """
        Change the settings for
        """
        self.user_id = user_id                              # user id
        self.nchannels = nchannels                          # number of channels
        self.channel_labels = channel_labels                # EEG electrode placements
        self.fsample = fsample                              # sampling rate
        self.max_size = max_size                            # maximum size of eeg
        self.explicit_settings = True                       # settings are explicit and will not be updated based on headset data


        if(len(channel_labels) != self.nchannels):
            print("Channel locations do not fit number of channels!!!")
            self.channel_labels = ['?'] * self.nchannels

    # Load data from a variety of sources
    # Currently only suports .xdf format
    def load_offline_eeg_data(self, filename, format='xdf', subset=[], print_output=True):
        """
        Loads offline data from a file

        Currently only supports .xdf
        """
        self.subset = subset

        if(format == 'xdf'):
            if print_output:
                print("loading ERP data from {}".format(filename))

            # load from xdf
            data, self.header = pyxdf.load_xdf(filename)

            # get the indexes of data
            for i in range(len(data)):
                namestring = data[i]['info']['name'][0]
                typestring = data[i]['info']['type'][0]
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
                self.marker_data = data[self.marker_index]['time_series']
                self.marker_timestamps = data[self.marker_index]['time_stamps']
            except: 
                print("Marker data not available")

            try:
                self.eeg_data = data[self.eeg_index]['time_series']
                self.eeg_timestamps = data[self.eeg_index]['time_stamps']
            except: 
                print("EEG data not available")

            try:
                self.response_data = data[self.response_index]['time_series']
                self.response_timestamps = data[self.response_index]['time_stamps']
            except: 
                print("Response data not available")

            # Unless explicit settings are desired, get settings from headset
            #if self.explicit_settings == False:
            self.get_info_from_file(data, print_output)



        # support for other file types goes here

        # otherwise show an error
        else:
            print("Error: file format not supported")

    # Get metadata saved to the offline data file to fill in headset information
    def get_info_from_file(self, data, print_output=True):
        """
        Get EEG metadata from the stream

        Parameters:
        self    -   bci_data instance
        data    -   data object from the LSL stream

        Returns:
        self    -   bci_data instance

        """

        self.headset_string = data[self.eeg_index]['info']['name'][0]            # headset name in string format
        self.fsample = float(data[self.eeg_index]['info']['nominal_srate'][0])   # sampling rate
        self.nchannels = int(data[self.eeg_index]['info']['channel_count'][0])   # number of channels 
        
        #get chtypes, chunits
        self.ch_type = []
        self.ch_units = []
        for i in range(self.nchannels):
            # type
            ch_type = data[self.eeg_index]['info']["desc"][0]["channels"][0]["channel"][i]["type"][0]
            # send to lower case letters for mne
            ch_type = ch_type.lower()
            # save trigger channel as stim
            if ch_type == "trg":
                ch_type = 'stim'
            #add to list
            self.ch_type.append(ch_type)

            #units
            ch_units = data[self.eeg_index]['info']["desc"][0]["channels"][0]["channel"][i]["unit"][0]
            self.ch_units.append(ch_units)


        self.channel_labels = []                                                 # channel labels/locations, 'TRG' means trigger
        try:
            for i in range(self.nchannels):
                self.channel_labels.append(data[self.eeg_index]['info']['desc'][0]['channels'][0]['channel'][i]['label'][0])

        except:
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
            self.subset_indices = list(range(0,self.nchannels))

        # send channel labels to classifier
        try:
            self.classifier.channel_labels = self.channel_labels
        except:
            if print_output:
                print("no classifier defined")

        if print_output:
            print(self.headset_string)
            print(self.channel_labels)

    # ONLINE
    # stream data from an online source
    def stream_online_eeg_data(self, timeout=5, max_eeg_samples=1000000, max_marker_samples=100000, eeg_only=False, subset=[]):
        """
        Howdy
        """
        self.subset = subset

        print("printing incoming stream")

        self.subset = subset

        exit_flag = 0

        if eeg_only == False:
            try:
                print("Resolving LSL marker stream... ")
                marker_stream = resolve_byprop('type', 'LSL_Marker_Strings', timeout=timeout)
                self.marker_inlet = StreamInlet(marker_stream[0], processing_flags = 0)
                print("Getting stream info...")
                marker_info = self.marker_inlet.info()
                print("The marker stream's XML meta-data is: ")
                print(marker_info.as_xml())
                
            except Exception as e:
                print("No marker stream currently available")
                exit_flag = 1
            
        # Resolve EEG marker stream
        try: 
            print("Resolving LSL EEG stream... ")
            eeg_stream = resolve_byprop('type', 'EEG', timeout=timeout)
            self.eeg_inlet = StreamInlet(eeg_stream[0], processing_flags = 0)
            print("Getting stream info...")
            eeg_info = self.eeg_inlet.info()
            print("The EEG stream's XML meta-data is: ")
            print(eeg_info.as_xml())
            print(eeg_info)
            #print(eeg_info.created_at)

            # if there are no explicit settings
            if self.explicit_settings == False:
                self.get_info_from_stream()
            
        except Exception as e:
            print("No EEG stream currently available")
            print(e) # print the exception
            exit_flag = 1

        # Exit if one or both streams are unavailable
        if exit_flag == 1:
            print("No streams available, exiting...")
            sys.exit()

        self.marker_data = []
        self.marker_timestamps = []
        self.eeg_data = []
        self.eeg_timestamps = []
        
        self.marker_data = np.array(self.marker_data)
        self.marker_timestamps = np.array(self.marker_timestamps)
        self.eeg_data = np.array(self.eeg_data)
        self.eeg_timestamps = np.array(self.eeg_timestamps)
    
    # Get headset data from stream
    def get_info_from_stream(self):
        """
        Howdy
        """
        # get info obect from stream
        eeg_info = self.eeg_inlet.info()

        self.headset_string = eeg_info.name()            # headset name in string format
        self.fsample = float(eeg_info.nominal_srate())   # sampling rate
        self.nchannels = int(eeg_info.channel_count())   # number of channels 

        # get online channel types and units

        # iterate through children of <"channels"> to get the channel labels
        ch = eeg_info.desc().child("channels").child("channel")
        self.channel_labels = []                         # channel labels/locations, 'TRG' means trigger
        print("num channels = ", self.nchannels)
        for i in range(self.nchannels):
            name = ch.child_value("name")
            if name == "":
                name = ch.child_value("label")
            self.channel_labels.append(name)
            # go to next sibling
            ch = ch.next_sibling()

        # if it is the DSI7 flex, relabel the channels, may want to make this more flexible in the future
        if self.headset_string == "DSI7":  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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


        # If a subset is to be used, define a new nchannels, and channel labels
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
            self.subset_indices = list(range(0,self.nchannels))

        # send channel labels to classifier
        try:
            self.classifier.channel_labels = self.channel_labels
        except:
            print("no classifier defined")

        # Print some headset info
        print(self.headset_string)
        print(self.channel_labels)

    # Get new data from stream
    def pull_data_from_stream(self, include_markers=True, include_eeg=True, return_eeg=False):
        """
        Howdy
        """
        # Pull chunks of new data
        if include_markers == True:
            # pull marker chunk
            new_marker_data, new_marker_timestamps = self.marker_inlet.pull_chunk(timeout=0.1)

            # pull marker time correction
            self.marker_time_correction = self.marker_inlet.time_correction()

            # apply time correction
            new_marker_timestamps = [new_marker_timestamps[i] + self.marker_time_correction for i in range(len(new_marker_timestamps))]

            # save the marker data to the data object
            self.marker_data = np.array(list(self.marker_data) + new_marker_data)
            self.marker_timestamps = np.array(list(self.marker_timestamps) + new_marker_timestamps)

        if include_eeg == True:
            new_eeg_data, new_eeg_timestamps = self.eeg_inlet.pull_chunk(timeout=0.1)
            new_eeg_data = np.array(new_eeg_data)
            
            #Handle the case when you are using subsets
            if self.subset!=[]:
                new_eeg_data = new_eeg_data[:, self.subset_indices]   

            # if time is in milliseconds, divide by 1000, works for sampling rates above 10Hz
            try:
                if self.time_units == 'milliseconds':
                    new_eeg_timestamps = [(new_eeg_timestamps[i]/1000) for i in range(len(new_eeg_timestamps))]

            # If time units are not defined then define them
            except:
                dif_low = -2
                dif_high = -1
                while (new_eeg_timestamps[dif_high] - new_eeg_timestamps[dif_low] == 0):
                    dif_low -= 1
                    dif_high -= 1
                
                if new_eeg_timestamps[dif_high] - new_eeg_timestamps[dif_low] > 0.1:
                    new_eeg_timestamps = [(new_eeg_timestamps[i]/1000) for i in range(len(new_eeg_timestamps))]
                    self.time_units = 'milliseconds'
                else: 
                    self.time_units = 'seconds'

            # apply time correction, this is essential for headsets like neurosity which have their own clock
            self.eeg_time_correction = self.eeg_inlet.time_correction()

            # MAYBE DONT NEED THIS WITH NEW PROC SETTINGS
            new_eeg_timestamps = [new_eeg_timestamps[i] + self.eeg_time_correction for i in range(len(new_eeg_timestamps))]
        
            # save the EEG data to the data object
            try:
                self.eeg_data = np.concatenate((self.eeg_data, new_eeg_data))
            except:
                self.eeg_data = new_eeg_data

            # save the marker data to the data object
            self.eeg_timestamps = np.array(list(self.eeg_timestamps) + new_eeg_timestamps)

        # If the outlet exists send a ping
        if self.stream_outlet == True:
            self.ping_count += 1
            if self.ping_count % self.ping_interval:
                self.outlet.push_sample(["ping"])

        # Return eeg
        if return_eeg == True:
            return new_eeg_timestamps, new_eeg_data

    def save_data(self, directory_name):
        """
        Save the data from different stages

        Creates a directory with x files
        data_pickle - includes raw EEG, markers, processed EEG, features


        """

    def mne_export_as_raw(self):
        """
        MNE Export

        Exports the EEG data as an epoch object

        * Requires MNE

        """
        print("mne_export_as_raw has not been implemented yet")
        # Check for mne
        try:
            import mne
        except:
            print("Could not import mne, you may have to install (pip install mne)")

        # create info from metadata
        info = mne.create_info(ch_names=self.channel_labels, sfreq=self.fsample, ch_types='eeg')

        # create the MNE epochs, pass in the raw

        # make sure that units match
        raw_data = self.eeg_data.transpose()
        raw_array = mne.io.RawArray(data=raw_data, info=info)

        # change the last column of epochs array events to be the class labels
        # raw_array.events[:, -1] = self.labels

        return raw_array


    def mne_export_as_epochs(self):
        """
        MNE Export

        Exports the EEG data as an epoch object

        * Requires MNE

        """

        # Check for mne
        try:
            import mne
        except:
            print("Could not import mne, you may have to install (pip install mne)")

        # create info from metadata
        info = mne.create_info(ch_names=self.channel_labels, sfreq=self.fsample, ch_types=self.ch_type)

        # create the MNE epochs, pass in the raw


        # make sure that units match
        epoch_data = self.raw_eeg_windows.copy()
        for i, u in enumerate(self.ch_units):
            if u == "microvolts":
                # convert to volts
                epoch_data[:,i,:] = epoch_data[:,i,:] / 1000000

        epochs_array = mne.EpochsArray(data=epoch_data, info=info)

        # change the last column of epochs array events to be the class labels
        epochs_array.events[:, -1] = self.labels

        return epochs_array

    def mne_export_resting_state_as_raw(self):
        """
        MNE Export

        Exports the EEG data as an epoch object

        * Requires MNE

        """
        print("mne_export_as_raw has not been implemented yet")
        # Check for mne
        try:
            import mne
        except:
            print("Could not import mne, you may have to install (pip install mne)")

        # create info from metadata
        info = mne.create_info(ch_names=self.channel_labels, sfreq=self.fsample, ch_types='eeg')


        try:
            # create the MNE epochs, pass in the raw

            # make sure that units match
            raw_data = self.rest_windows[0,:,:]
            raw_array = mne.io.RawArray(data=raw_data, info=info)

            # change the last column of epochs array events to be the class labels
            # raw_array.events[:, -1] = self.labels

        except:
            # could not find resting state data, sending the whole collection instead
            print("NO PROPER RESTING STATE DATA FOUND, SENDING ALL OF THE EEG DATA INSTEAD")
            raw_data = self.eeg_data.transpose()
            raw_array = mne.io.RawArray(data=raw_data, info=info)
        
        return raw_array


    # SIGNAL PROCESSING
    # Preprocessing goes here (windows are nchannels by nsamples)
    def preprocessing(self, window, option=None, order=5, fc=60, fl=10, fh=50):
        """
        Howdy
        """
        # do nothing
        if option == None:
            new_window = window
            return new_window    

        if option == 'notch':
            new_window = notchfilt(window, self.fsample, Q=30, fc=60)
            return new_window

        if option == 'bandpass':

            new_window = bandpass(window, fl, fh, order, self.fsample)
            return new_window

        # other preprocessing options go here

    # Artefact rejection goes here (windows are nchannels by nsamples)
    def artefact_rejection(self, window, option=None):
        """
        Howdy
        """
        # do nothing
        if option == None:
            new_window = window
            return new_window

        # other preprocessing options go here\

    def package_resting_state_data(self):
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
                    #print("received eyes open start")

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
                self.eyes_open_windows = np.ndarray((len(eyes_open_start_time), self.nchannels, nsamples))
                # Now copy EEG for these windows
                for i in range(len(eyes_open_start_time)):

                    # For each channel of the EEG, interpolate to uniform sampling rate
                    for c in range(self.nchannels):
                        # First, adjust the EEG timestamps to start from zero
                        eeg_timestamps_adjusted = self.eeg_timestamps[eyes_open_start_loc[i]:eyes_open_end_loc[i]] - self.eeg_timestamps[eyes_open_start_loc[i]]
                        
                        # Second, interpolate to timestamps at a uniform sampling rate
                        channel_data = np.interp(self.eyes_open_timestamps, eeg_timestamps_adjusted, self.eeg_data[eyes_open_start_loc[i]:eyes_open_end_loc[i],c])

                        # Third, add to the EEG window
                        self.eyes_open_windows[i,c,:] = channel_data
                        self.eyes_open_timestamps

            print("Done packaging resting state data")


            # Eyes closed 

            if len(eyes_closed_end_loc) > 0:
                # Get duration, nsmaples
                duration = np.floor(eyes_closed_end_time[0] - eyes_closed_start_time[0])
                nsamples = int(duration * self.fsample) 

                self.eyes_closed_timestamps = np.array(range(nsamples)) / self.fsample
                self.eyes_closed_windows = np.ndarray((len(eyes_closed_start_time), self.nchannels, nsamples))
                # Now copy EEG for these windows
                for i in range(len(eyes_closed_start_time)):

                    # For each channel of the EEG, interpolate to uniform sampling rate
                    for c in range(self.nchannels):
                        # First, adjust the EEG timestamps to start from zero
                        eeg_timestamps_adjusted = self.eeg_timestamps[eyes_closed_start_loc[i]:eyes_closed_end_loc[i]] - self.eeg_timestamps[eyes_closed_start_loc[i]]
                        
                        # Second, interpolate to timestamps at a uniform sampling rate
                        channel_data = np.interp(self.eyes_closed_timestamps, eeg_timestamps_adjusted, self.eeg_data[eyes_closed_start_loc[i]:eyes_closed_end_loc[i],c])

                        # Third, add to the EEG window
                        self.eyes_closed_windows[i,c,:] = channel_data
                        self.eyes_closed_timestamps

            # Rest
            if len(rest_end_loc) > 0:
                # Get duration, nsmaples
                while(rest_end_time[0] < rest_start_time[0]):
                    rest_end_time.pop(0)
                    rest_end_loc.pop(0)

                duration = np.floor(rest_end_time[0] - rest_start_time[0])
                

                nsamples = int(duration * self.fsample) 

                self.rest_timestamps = np.array(range(nsamples)) / self.fsample
                self.rest_windows = np.ndarray((len(rest_start_time), self.nchannels, nsamples))
                # Now copy EEG for these windows
                for i in range(len(rest_start_time)):

                    # For each channel of the EEG, interpolate to uniform sampling rate
                    for c in range(self.nchannels):
                        # First, adjust the EEG timestamps to start from zero
                        eeg_timestamps_adjusted = self.eeg_timestamps[rest_start_loc[i]:rest_end_loc[i]] - self.eeg_timestamps[rest_start_loc[i]]
                        
                        # Second, interpolate to timestamps at a uniform sampling rate
                        channel_data = np.interp(self.rest_timestamps, eeg_timestamps_adjusted, self.eeg_data[rest_start_loc[i]:rest_end_loc[i],c])

                        # Third, add to the EEG window
                        self.rest_windows[i,c,:] = channel_data
                        self.rest_timestamps
        except:
            print("Failed to package resting state data")


    # main
    # add pp_low, pp_high, pp_order, subset

    def main(self, 
            buffer=0.01, 
            eeg_start=0, 
            max_channels = 64, 
            max_samples = 2560, 
            max_windows = 1000,
            max_loops=1000000, 
            training=True, 
            online=True, 
            train_complete=False, 
            iterative_training = False, 
            live_update = False,
            print_markers = True,
            print_training =True,
            print_fit=True,
            print_performance=True,
            print_predict=True,
            
            pp_type = "bandpass",   # preprocessing method
            pp_low=1,               # bandpass lower cutoff
            pp_high=40,             # bandpass upper cutoff
            pp_order=5              # bandpass order
            ):

        """
        Howdy
        """
        # Check if there is a classifier defined
        try:
            clf = self.classifier
            self.classifier_defined = True
        except:
            self.classifier_defined = False

        # flag that classifier has not yet been defined, if false it means it will be defined by the marker data

        # if this is the first time this function is being called for a given dataset then run some initialization
        if eeg_start == 0:
            self.window_end_buffer = buffer
            search_index = 0

            # initialize windows and labels
            current_raw_eeg_windows = np.zeros((max_windows,max_channels,max_samples))
            current_processed_eeg_windows = current_raw_eeg_windows
            current_labels = np.zeros((max_windows))

            self.raw_eeg_windows = np.zeros((max_windows,max_channels,max_samples))
            self.processed_eeg_windows = self.raw_eeg_windows
            self.labels = np.zeros((max_windows))               # temporary labels
            self.training_labels = np.zeros((max_windows))      # permanent training labels


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
            if online == False:
                loops = max_loops

            # if online, then pull new data with each iteration
            if online == True:
                self.pull_data_from_stream()

                # Create a stream to send markers back to Unity, but only create the stream once
                if self.stream_outlet == False:
                    # define the stream information
                    info = StreamInfo(name='PythonResponse', type='BCI', channel_count=1, nominal_srate=IRREGULAR_RATE, channel_format='string', source_id='pyp30042')
                    #print(info)
                    # create the outlet
                    self.outlet = StreamOutlet(info)
                    
                    # next make an outlet
                    print("the outlet exists")
                    self.stream_outlet = True

                    # Push the data
                    self.outlet.push_sample(["This is the python response stream"])
            
            # check if there is an available marker, if not, break and wait for more data
            while(len(self.marker_timestamps) > self.marker_count):
                loops = 0

                # If the marker contains a single string, not including ',' and begining with a alpha character, then it is an event message
                if len(self.marker_data[self.marker_count][0].split(',')) == 1 and self.marker_data[self.marker_count][0][0].isalpha():
                    # send feedback to unity if there is an available outlet
                    if self.stream_outlet == True:
                        # send feedback for each marker that you receive
                        self.outlet.push_sample(["marker received : {}".format(self.marker_data[self.marker_count][0])])

                    ############
                    if print_markers == True:
                        print(self.marker_data[self.marker_count][0])

                    # once all resting state data is collected then go and compile it
                    if self.marker_data[self.marker_count][0] == "Done with all RS collection":
                        self.package_resting_state_data()
                        self.marker_count += 1

                    elif self.marker_data[self.marker_count][0] == 'Trial Started':
                        if print_markers == True:
                            print("Trial started")
                        # Note that a marker occured, but do nothing else
                        self.marker_count += 1

                    elif self.marker_data[self.marker_count][0] == 'Trial Ends':
                        if print_markers == True:
                            print("Trial ended")

                        # If no classifier, then ideally just continue adding to the windows and labels arrays
                        if self.classifier_defined == False:
                            print("NO CLASSIFIER DEFINED")
                            self.marker_count += 1
                            break

                        # Trim the unused ends of numpy arrays
                        current_raw_eeg_windows = current_raw_eeg_windows[0:current_nwindows, 0:self.nchannels, 0:self.nsamples]
                        current_processed_eeg_windows = current_processed_eeg_windows[0:current_nwindows, 0:self.nchannels, 0:self.nsamples]
                        current_labels = current_labels[0:current_nwindows]

                        # TRAIN
                        if training == True:
                            self.classifier.add_to_train(current_processed_eeg_windows, current_labels, print_training=print_training)

                            if print_training:
                                print(current_nwindows, " windows and labels added to training set")

                            # if iterative training is on and active then also make a prediction
                            if iterative_training == True:
                                if print_predict:
                                    print("Added current samples to training set, now making a prediction")
                                prediction = self.classifier.predict(current_processed_eeg_windows, print_predict=print_predict)

                                # Send the prediction to Unity
                                if print_predict:
                                    print("{} was selected by the iterative classifier, sending to Unity".format(prediction))
                                # pick a sample to send an wait for a bit
                                
                                # if online, send the packet to Unity
                                if online == True:
                                    self.outlet.push_sample(["{}".format(prediction)])
                        
                        # PREDICT
                        elif train_complete == True and current_nwindows != 0:
                            if print_predict:
                                print("making a prediction based on ", current_nwindows ," windows")

                            if current_nwindows == 0:
                                print("No windows to make a decision")
                                self.marker_count += 1
                                break
                            
                            # save the online selection indices
                            selection_inds = list(range(self.nwindows - current_nwindows, self.nwindows))
                            self.online_selection_indices.append(selection_inds)

                            # make the prediciton
                            try:
                                prediction = self.classifier.predict(current_processed_eeg_windows, print_predict)
                                self.online_selections.append(prediction)
                            

                                if print_predict:
                                    print("Recieved prediction from classifier")

                                    # Send the prediction to Unity
                                    print("{} was selected, sending to Unity".format(prediction))
                                
                                # if online, send the packet to Unity
                                if online == True:
                                    self.outlet.push_sample(["{}".format(prediction)])

                            except:
                                if print_predict:
                                    print("This classification failed...")

                        # OH DEAR
                        else:
                            print("Unable to classify... womp womp")
                        
                        # Reset windows and labels
                        self.marker_count += 1
                        current_nwindows = 0
                        current_raw_eeg_windows = np.zeros((max_windows,max_channels,max_samples))
                        current_processed_eeg_windows = current_raw_eeg_windows
                        current_labels = np.zeros((max_windows))

                    # If training completed then train the classifier
                    elif self.marker_data[self.marker_count][0] == 'Training Complete' and train_complete == False:
                        if self.classifier_defined == False:
                            print("NO CLASSIFIER DEFINED")
                            self.marker_count += 1
                            break
                        if print_training:
                            print("Training the classifier")

                        self.classifier.fit(print_fit = print_fit, print_performance=print_performance)
                        train_complete = True
                        training = False
                        self.marker_count += 1

                    elif self.marker_data[self.marker_count][0] == 'Update Classifier':
                        if print_training:
                            print("Retraining the classifier")

                        self.classifier.fit(print_fit = print_fit, print_performance=print_performance)

                        iterative_training = True
                        if online == True:
                            live_update = True

                        self.marker_count += 1

                    else:
                        self.marker_count += 1

                    if online:
                        time.sleep(0.01)
                    loops += 1
                    continue
  
                
                # Get marker info
                marker_info = self.marker_data[self.marker_count][0].split(',')

                self.paradigm_string = marker_info[0]
                self.num_options = int(marker_info[1])
                label = int(marker_info[2])              
                self.window_length = float(marker_info[3])      # window length
                if len(marker_info) > 4:                        # if longer, collect this info and maybe it can be used by the classifier
                    self.meta = []
                    for i in range(4,len(marker_info)):
                        self.meta.__add__([marker_info[i]])

                    # Load the correct SSVEP freqs
                    if marker_info[0] == "ssvep":
                        clf.target_freqs = [1] * (len(marker_info) - 4)
                        clf.sampling_freq = self.fsample
                        for i in range(4,len(marker_info)):

                            clf.target_freqs[i - 4] = float(marker_info[i])
                            # print("changed ", i-4, "target frequency to", marker_info[i])

                # Check if the whole EEG window corresponding to the marker is available
                end_time_plus_buffer = self.marker_timestamps[self.marker_count] + self.window_length + buffer

                # If we don't have the full window then pull more data, only do this online
                if (self.eeg_timestamps[-1] <= end_time_plus_buffer):
                    if online == True:
                        break
                    if online == False:
                        self.marker_count += 1
                        break

                if print_markers == True:
                    print(marker_info)

                # send feedback to unity if there is an available outlet
                if self.stream_outlet == True:
                    print("sending feedback to Unity")
                    # send feedback for each marker that you receive
                    self.outlet.push_sample(["marker received : {}".format(self.marker_data[self.marker_count][0])])                

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
                    eeg_timestamps_adjusted = self.eeg_timestamps[start_loc:end_loc] - self.eeg_timestamps[start_loc]
                    
                    # Second, interpolate to timestamps at a uniform sampling rate
                    channel_data = np.interp(self.window_timestamps, eeg_timestamps_adjusted, self.eeg_data[start_loc:end_loc,c])

                    # Third, sdd to the EEG window
                    current_raw_eeg_windows[current_nwindows,c,0:self.nsamples] = channel_data

                # This is where to do preprocessing
                current_processed_eeg_windows[current_nwindows,:self.nchannels,:self.nsamples] = self.preprocessing(window=current_raw_eeg_windows[current_nwindows,:self.nchannels,:self.nsamples],option=pp_type, order=pp_order, fl=pp_low, fh=pp_high)

                # This is where to do artefact rejection
                current_processed_eeg_windows[current_nwindows,:self.nchannels,:self.nsamples] = self.artefact_rejection(window=current_processed_eeg_windows[current_nwindows,:self.nchannels,:self.nsamples],option=None)
                
                # Add the label if it exists, otherwise set a flag of -1 to denote that there is no label
                # if training == True:
                #     current_labels[current_nwindows] = label
                # else:
                #     current_labels[current_nwindows] = -1
                current_labels[current_nwindows] = label

                # copy to the eeg_data object
                self.raw_eeg_windows[self.nwindows,0:self.nchannels,0:self.nsamples] = current_raw_eeg_windows[current_nwindows,0:self.nchannels,0:self.nsamples]
                self.processed_eeg_windows[self.nwindows,0:self.nchannels,0:self.nsamples] = current_processed_eeg_windows[current_nwindows,0:self.nchannels,0:self.nsamples]
                self.labels[self.nwindows] = current_labels[current_nwindows]

                # Send live updates
                if live_update == True:
                    try:
                        if self.nsamples != 0:
                            pred = self.classifier.predict(current_processed_eeg_windows[current_nwindows, 0:self.nchannels, 0:self.nsamples], print_predict=print_predict)
                            self.outlet.push_sample(["{}".format(int(pred[0]))])
                    except: 
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
        self.raw_eeg_windows = self.raw_eeg_windows[0:self.nwindows, 0:self.nchannels, 0:self.nsamples]
        self.processed_eeg_windows = self.processed_eeg_windows[0:self.nwindows, 0:self.nchannels, 0:self.nsamples]
        self.labels = self.labels[0:self.nwindows]
        #self.predictions = self.predictions[0:self.nwindows]

# ERP Data
class ERP_data(EEG_data):
    """
    Howdy
    """
    def mne_export_as_raw(self):
        """
        MNE Export

        Exports the EEG data as an epoch object

        * Requires MNE

        """
        print("mne_export_as_raw has not been implemented yet")
        # # Check for mne
        # try:
        #     import mne
        # except:
        #     print("Could not import mne, you may have to install (pip install mne)")

        # # create info from metadata
        # info = mne.create_info(ch_names=self.channel_labels, sfreq=self.fsample, ch_types='eeg')

        # # create the MNE epochs, pass in the raw

        # # make sure that units match
        # epochs_array = mne.EpochsArray(data=self.raw_eeg_windows, info=info)

        # # change the last column of epochs array events to be the class labels
        # epochs_array.events[:, -1] = self.labels

        # return epochs_array


    def mne_export_as_epochs(self):
        """
        MNE Export

        Exports the ERP data as an epoch object

        * Requires MNE

        """

        # Check for mne
        try:
            import mne
        except:
            print("Could not import mne, you may have to install (pip install mne)")

        # create info from metadata
        info = mne.create_info(ch_names=self.channel_labels, sfreq=self.fsample, ch_types=self.ch_type)

        # create the MNE epochs, pass in the raw


        # make sure that units match
        epoch_data = self.erp_windows_processed[:len(self.target_index),:,:].copy()
        for i, u in enumerate(self.ch_units):
            if u == "microvolts":
                # convert to volts
                epoch_data[:,i,:] = epoch_data[:,i,:] / 1000000

        epochs_array = mne.EpochsArray(data=epoch_data, info=info)

        # change the last column of epochs array events to be the class labels
        epochs_array.events[:, -1] = self.target_index.astype(int)

        return epochs_array

    def mne_export_as_evoked(self):
        """
        MNE Export

        Exports the EEG data as an epoch object

        * Requires MNE

        """
        print("mne_export_as_evoked has not yet been implemented")
        # # Check for mne
        # try:
        #     import mne
        # except:
        #     print("Could not import mne, you may have to install (pip install mne)")

        # # create info from metadata
        # info = mne.create_info(ch_names=self.channel_labels, sfreq=self.fsample, ch_types=self.ch_type)

        # # create the MNE epochs, pass in the raw


        # # make sure that units match
        # evoked_data = self.raw_eeg_windows.copy()
        # for i, u in enumerate(self.ch_units):
        #     if u == "microvolts":
        #         # convert to volts
        #         evoked_data[:,i,:] = evoked_data[:,i,:] / 1000000

        # evoked_array = mne.EpochsArray(data=evoked_data, info=info, tmin=self.window_start)

        # # change the last column of epochs array events to be the class labels
        # evoked_array.events[:, -1] = self.labels

        # return evoked_array

    # Formats the ERP data, call this every time that a new chunk arrives
    def main(self, 
            window_start=0.0, 
            window_end=0.8, 
            eeg_start=0, 
            buffer=0.01, 
            max_num_options=64,
            max_windows_per_option=50,
            max_windows=10000, 
            max_decisions=500, 
            max_loops=1000000000, 
            training=False, 
            online=False,
            print_markers=True,
            print_training=True,
            print_fit=True,
            print_performance=True,
            print_predict=True,


            # Preprocessing
            pp_type = "bandpass",   # preprocessing method
            pp_low=1,               # bandpass lower cutoff
            pp_high=40,             # bandpass upper cutoff
            pp_order=5,             # bandpass order

            plot_erp = False
            ):
        """
        Howdy
        """

        unity_train = True
        self.num_options = max_num_options

        # plot settings
        self.plot_erp = plot_erp
        if self.plot_erp == True:
            fig1, axs1 = plt.subplots(self.nchannels)
            fig2, axs2 = plt.subplots(self.nchannels)
            non_target_plot = 99


        # iff this is the first time this function is being called for a given dataset
        if eeg_start == 0:
            self.window_size = window_end - window_start
            self.nsamples = int(np.ceil(self.window_size * self.fsample) + 1)
            self.window_end_buffer = buffer
            self.num_options = max_num_options
            self.max_windows = max_windows
            self.max_windows_per_option = max_windows_per_option
            self.max_decisions = max_decisions

            search_index = 0

            self.window_timestamps = np.arange(self.nsamples) / self.fsample
            
            # initialize the numbers of markers, windows, and decision blocks to zero
            self.marker_count = 0
            self.nwindows = 0
            self.decision_count = 0


            self.training_labels= np.zeros((self.max_windows), dtype=int)
            self.stim_labels = np.zeros((self.max_windows, self.num_options), dtype=bool)
            self.target_index = np.ndarray((self.max_windows), bool)
            
            # initialize the data structures in numpy arrays
            # ERP windows
            self.erp_windows_raw = np.zeros((self.max_windows, self.nchannels, self.nsamples))
            self.erp_windows_processed = np.zeros((self.max_windows, self.nchannels, self.nsamples))

            # Windows per decision, ie. the number of times each stimulus has flashed
            self.windows_per_decision = np.zeros((self.num_options))

            # Decision blocks are the ensemble averages of all windows collected for each stimulus object
            self.decision_blocks_raw = np.ndarray((self.max_decisions, self.num_options, self.nchannels, self.nsamples))
            self.decision_blocks_processed = np.ndarray((self.max_decisions, self.num_options, self.nchannels, self.nsamples))

            # Big decision blocks contain all decisions, all stimulus objects, all windows, all channels, and all samples (they are BIG)
            self.big_decision_blocks_raw = np.ndarray((self.max_decisions, self.num_options, self.max_windows_per_option, self.nchannels, self.nsamples))
            self.big_decision_blocks_processed = np.ndarray((self.max_decisions, self.num_options, self.max_windows_per_option, self.nchannels, self.nsamples))

            # Initialize the
            self.num_options_per_decision = np.zeros((max_decisions))

            loops = 0
            train_complete = False 
        
        while loops < max_loops:
            # load data chunk from search start position
            # offline no more data to load
            if online == False:
                loops = max_loops

            # online load data
            if online == True:
                # Time sync if not synced
                
                self.pull_data_from_stream()

                # Create a stream to send markers back to Unity, but only create the stream once
                if self.stream_outlet == False:
                    # define the stream information
                    info = StreamInfo(name='PythonResponse', type='BCI', channel_count=1, nominal_srate=IRREGULAR_RATE, channel_format='string', source_id='pyp30042')
                    #print(info)
                    # create the outlet
                    self.outlet = StreamOutlet(info)
                    
                    # next make an outlet
                    print("the outlet exists")
                    self.stream_outlet = True

                    # Push the data
                    self.outlet.push_sample(["This is the python response stream"])

            # check if there is an available marker, if not, break and wait for more data
            while(len(self.marker_timestamps) > self.marker_count):
                loops = 0
                if len(self.marker_data[self.marker_count][0].split(',')) == 1:

                    # if self.marker_data[self.marker_count][0] == 'P300 SingleFlash Begins' or 'P300 SingleFlash Started':
                    if self.marker_data[self.marker_count][0] == 'P300 SingleFlash Started' or self.marker_data[self.marker_count][0] == 'P300 SingleFlash Begins' or self.marker_data[self.marker_count][0] == 'Trial Started':
                        # Note that a marker occured, but do nothing else
                        if print_markers == True:
                            print("Trial Started")
                        self.marker_count += 1

                    # once all resting state data is collected then go and compile it
                    elif self.marker_data[self.marker_count][0] == "Done with all RS collection":
                        self.package_resting_state_data()
                        self.marker_count += 1


                    # If training completed then train the classifier
                    elif self.marker_data[self.marker_count][0] == 'Training Complete' and train_complete == False:
                        if train_complete == False:
                            if print_training:
                                print("Training the classifier")
                            self.classifier.fit(print_fit = print_fit, print_performance=print_performance)
                        train_complete = True
                        training = False
                        self.marker_count += 1
                        #continue

                    # if there is a P300 end flag increment the decision_index by one
                    elif self.marker_data[self.marker_count][0] == 'P300 SingleFlash Ends' or self.marker_data[self.marker_count][0] == 'Trial Ends':

                        # get the smallest number of windows per decision in the case the are not the same
                        num_ensemble_windows = int(np.min(self.windows_per_decision))

                        # save the number of options
                        self.num_options_per_decision[self.decision_count] = int(self.num_options)

                        # Raw ensemble average
                        ensemble_average_block = np.mean(self.big_decision_blocks_raw[self.decision_count, 0:self.num_options, 0:num_ensemble_windows, 0:self.nchannels, 0:self.nsamples], axis=1)
                        self.decision_blocks_raw[self.decision_count, 0:self.num_options, 0:self.nchannels, 0:self.nsamples] = ensemble_average_block

                        # Processed ensemble average
                        ensemble_average_block = np.mean(self.big_decision_blocks_processed[self.decision_count, 0:self.num_options, 0:num_ensemble_windows, 0:self.nchannels, 0:self.nsamples], axis=1)
                        self.decision_blocks_processed[self.decision_count, 0:self.num_options, 0:self.nchannels, 0:self.nsamples] = ensemble_average_block

                        # Reset windows per decision
                        self.windows_per_decision = np.zeros((self.num_options))

                        if self.plot_erp == True:
                            fig1.show()
                            fig2.show()

                        if print_markers == True:
                            print(self.marker_data[self.marker_count][0])

                        self.marker_count += 1

                        # ADD IN A CHECK TO MAKE SURE THERE IS SUFFICIENT INFO TO MAKE A DECISION
                        # IF NOT THEN MOVE ON TO THE NEXT ONE
                        if True:
                            # CLASSIFICATION
                            # if the decision block has a label then add to training set
                            #if training == True:
                            #if self.decision_count <= len(self.labels) - 1:
                            if train_complete == False:
                                # ADD to training set
                                if unity_train == True:
                                    if print_training:
                                        print("adding decision block {} to the classifier with label {}".format(self.decision_count, unity_label))
                                    self.classifier.add_to_train(self.decision_blocks_processed[self.decision_count,:self.num_options,:,:], unity_label, print_training=print_training)

                                    # plot what was added
                                    #decision_vis(self.decision_blocks[self.decision_count,:,:,:], self.fsample, unity_label, self.channel_labels)
                                else:
                                    if print_training:
                                        print("adding decision block {} to the classifier with label {}".format(self.decision_count, self.labels[self.decision_count]))
                                    self.classifier.add_to_train(self.decision_blocks_processed[self.decision_count,:self.num_options,:,:], self.labels[self.decision_count], print_train=print_training)

                                    # if the last of the labelled data was just added
                                    if self.decision_count == len(self.labels) - 1:

                                        # FIT
                                        print("training the classifier")
                                        self.classifier.fit(n_splits=len(self.labels), print_fit = print_fit, print_performance=print_performance)

                            # else do the predict the label
                            else:
                                # PREDICT
                                prediction = self.classifier.predict_decision_block(decision_block=self.decision_blocks_processed[self.decision_count,0:self.num_options,:,:], print_predict=print_predict)    

                                # save the selection indices                  

                                # Send the prediction to Unity
                                if print_predict:
                                    print("{} was selected, sending to Unity".format(prediction))
                                # pick a sample to send an wait for a bit
                                
                                # if online, send the packet to Unity
                                if online == True:
                                    self.outlet.push_sample(["{}".format(prediction)])
                    
                        # TODO
                        else:
                            print("Insufficient windows to make a decision")
                            self.decision_count -= 1
                            #print(self.windows_per_decision)

                        self.decision_count += 1
                        self.windows_per_decision = np.zeros((self.num_options))

                        # UPDATE THE SEARCH START LOC
                        #continue
                    else:
                        self.marker_count += 1

                    if online:
                        time.sleep(0.01)
                    loops += 1
                    continue

                # Check if the whole EEG window corresponding to the marker is available
                end_time_plus_buffer = self.marker_timestamps[self.marker_count] + window_end + buffer
                
                if (self.eeg_timestamps[-1] <= end_time_plus_buffer):
                    # UPDATE THE SEARCH START LOC
                    break

                if online == True:
                    self.outlet.push_sample(["python got marker: {}".format(self.marker_data[self.marker_count][0])])

                # If the whole EEG is available then add it to the erp window and the decision block
                
                # Markers are in the format [p300, single (s) or multi (m),num_selections, train_target_index, flash_index_1, flash_index_2, ... ,flash_index_n]

                # Get marker info
                marker_info = self.marker_data[self.marker_count][0].split(',')

                # unity_flash_indexes 
                flash_indices = list()


                for i, info in enumerate(marker_info):
                    if i == 0:
                        bci_string = info
                    elif i == 1:
                        self.flash_type = info
                    elif i == 2:
                        # If there is a different number of options 
                        if self.num_options != int(info):
                            self.num_options = int(info)

                            # Resize on the first marker
                            self.windows_per_decision = np.zeros((self.num_options))
                    elif i == 3:
                        unity_label = int(info)
                    elif i >= 4:
                        flash_indices.append(int(info))

                self.windows_per_decision[flash_indices] += 1

                # During training, 
                # should this be repeated for multiple flash indices
                # for flash_index in flash_indices:
                if training == True:
                    # Get target info
                    
                    #current_target = target_order[self.decision_count]
                    if unity_train == True:
                        if print_markers == True:
                            print(marker_info)
                        current_target = unity_label

                    self.training_labels[self.nwindows] = current_target

                    for fi in flash_indices:
                        self.stim_labels[self.nwindows, fi] = True

                    if current_target in flash_indices:
                        self.target_index[self.nwindows] = True
                    else:
                        self.target_index[self.nwindows] = False

                # Find the start time and end time for the window based on the marker timestamp
                start_time = self.marker_timestamps[self.marker_count] + window_start
                end_time = self.marker_timestamps[self.marker_count] + window_end

                # locate the indices of the window in the eeg data
                for i, s in enumerate(self.eeg_timestamps[search_index:-1]):
                    #print("i,s",i,s)
                    if s > start_time:
                        start_loc = search_index + i - 1
                        # if start_loc < 0:
                        #     start_loc = 0

                        break
                end_loc = start_loc + self.nsamples + 1

                # Adjust windows per option
                #self.windows_per_option = np.zeros(self.num_options, dtype=int)

                #print("start loc, end loc ", start_loc, end_loc)
                # linear interpolation and add to numpy array
                for flash_index in flash_indices:
                    for c in range(self.nchannels):
                        eeg_timestamps_adjusted = self.eeg_timestamps[start_loc:end_loc] - self.eeg_timestamps[start_loc]

                        channel_data = np.interp(self.window_timestamps, eeg_timestamps_adjusted, self.eeg_data[start_loc:end_loc,c])

                        # add to raw ERP windows
                        self.erp_windows_raw[self.nwindows, c, 0:self.nsamples] = channel_data
                        #self.decision_blocks_raw[self.decision_count, self.nwindows, c, 0:self.nsamples]

                        # if pp_type == "bandpass":
                        #     channel_data_2 = bandpass(channel_data[np.newaxis,:], pp_low, pp_high, pp_order, self.fsample)
                        #     channel_data = channel_data_2[0,:]

                        # # Add to the instance count
                        # self.windows_per_decision[flash_index] += 1

                        if self.plot_erp == True:
                            if flash_index == current_target:
                                axs1[c].plot(range(self.nsamples),channel_data)

                            elif non_target_plot == 99 or non_target_plot == flash_index:
                                axs2[c].plot(range(self.nsamples),channel_data)
                                non_target_plot = flash_index

                        # # add to processed ERP windows
                        # self.erp_windows[self.nwindows, c, 0:self.nsamples] = channel_data

                        # # Does the ensemble avearging
                        # self.decision_blocks[self.decision_count, flash_index, c, 0:self.nsamples] += channel_data

                    # This is where to do preprocessing
                    self.erp_windows_processed[self.nwindows,:self.nchannels,:self.nsamples] = self.preprocessing(window=self.erp_windows_raw[self.nwindows,:self.nchannels,:self.nsamples],option=pp_type, order=pp_order, fl=pp_low, fh=pp_high)

                    # This is where to do artefact rejection
                    self.erp_windows_processed[self.nwindows,:self.nchannels,:self.nsamples] = self.artefact_rejection(window=self.erp_windows_processed[self.nwindows,:self.nchannels,:self.nsamples],option=None)

                    # Add the raw window to the raw decision blocks
                    #self.decision_blocks_raw[self.decision_count, flash_index, 0:self.nchannels, 0:self.nsamples] +=  self.erp_windows_processed
                    self.big_decision_blocks_raw[self.decision_count, flash_index, int(self.windows_per_decision[flash_index] - 1), 0:self.nchannels, 0:self.nsamples] = self.erp_windows_raw[self.nwindows,:self.nchannels,:self.nsamples]

                    self.big_decision_blocks_processed[self.decision_count, flash_index, int(self.windows_per_decision[flash_index] - 1), 0:self.nchannels, 0:self.nsamples] = self.erp_windows_processed[self.nwindows,:self.nchannels,:self.nsamples]
                    
                    # self.windows_per_decision[flash_index] += 1
                # Reset for the next decision


                # iterate to next window
                self.marker_count += 1
                self.nwindows += 1
                search_index = start_loc
                if online:
                    time.sleep(0.000001)

            if online:
                time.sleep(0.000001)
            loops += 1
            
        # Trim the unused ends of numpy arrays
        if training == True:
            self.training_labels = self.training_labels[0:self.nwindows-1]
            self.target_index = self.target_index[0:self.nwindows-1]

        # self.erp_windows = self.erp_windows[0:self.nwindows, 0:self.nchannels, 0:self.nsamples]
        self.erp_windows_raw = self.erp_windows_raw[0:self.nwindows, 0:self.nchannels, 0:self.nsamples]
        self.target_index = self.target_index[0:self.nwindows]
        self.training_labels = self.training_labels[0:self.nwindows]
        self.stim_labels = self.stim_labels[0:self.nwindows, :]
        self.num_options_per_decision = self.num_options_per_decision[0:self.decision_count]
        self.decision_blocks_raw = self.decision_blocks_raw[0:self.decision_count, :, 0:self.nchannels, 0:self.nsamples]
        self.decision_blocks_processed = self.decision_blocks_processed[0:self.decision_count, :, 0:self.nchannels, 0:self.nsamples]
        self.big_decision_blocks_raw = self.big_decision_blocks_raw[0:self.decision_count, :, :, 0:self.nchannels, 0:self.nsamples]
        self.big_decision_blocks_processed = self.big_decision_blocks_processed[0:self.decision_count, :, :, 0:self.nchannels, 0:self.nsamples]