#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marker LSL sim simulates offline data as online data

Additional Arguments:
now (-n)            -   start the stream immediately
num_loops (int)     -   number of times to repeat the stream

ex.
>python marker_lsl_sim.py now 8
Immediately begins the simuted stream, repeats 8 times

Created on Wed Apr 21 10:26:44 2021

@author: brianirvine
"""
import os
import sys
import time
import datetime

from pylsl import StreamInfo, StreamOutlet

# # Add parent directory to path to access bci_essentials
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))

# Import local bci_essentials
from bci_essentials.bci_data import EEG_data

# check whether to start now, or at the next even minute to sync with other programs
start_now = False
try:
    arg1 = sys.argv[1]
    if arg1 == 'now' or arg1 == '-n':
        print('starting stream immediately')
        start_now = True
except:
    start_now = False

# add the ability to loop n times, default is 1
nloops = 1
try:
    nloops = int(sys.argv[2])
    print('repeating for ', nloops, ' loops')

except:
    nloops = 1

# Identify the file to simulate
filename = "examples/data/p300_example.xdf"

# Load the example EEG stream
eeg_stream = EEG_data()
eeg_stream.load_offline_eeg_data(filename)

# Get the data from that stream
marker_time_stamps = eeg_stream.marker_timestamps
marker_time_series = eeg_stream.marker_data
eeg_time_stamps = eeg_stream.eeg_timestamps
eeg_time_series = eeg_stream.eeg_data

# find the time range of the marker stream and delete EEG data out of this range
time_start = min(marker_time_stamps)
time_stop = max(marker_time_stamps)

eeg_keep_ind = [(eeg_time_stamps > time_start)&(eeg_time_stamps < time_stop)]
eeg_time_stamps = eeg_time_stamps[tuple(eeg_keep_ind)]
eeg_time_series = eeg_time_series[tuple(eeg_keep_ind)]

# estimate sampling rates
fs_marker = round(len(marker_time_stamps) / (time_stop - time_start))
fs_eeg = round(len(eeg_time_stamps) / (time_stop - time_start))

i = 0
info = StreamInfo('MockMarker', 'LSL_Marker_Strings', 1, fs_marker, 'string', 'mockmark1')
outlet = StreamOutlet(info)

if start_now == False:
    # publish to stream at the next rounded minute 
    now_time = datetime.datetime.now()
    print("Current time is ", now_time)
    seconds = (now_time - now_time.min).seconds
    microseconds = now_time.microsecond
    # // is a floor division, not a comment on following line:
    rounding = (seconds+60/2) // 60 * 60
    round_time = now_time + datetime.timedelta(0,60+rounding-seconds,-microseconds)
    print(microseconds)
    print("Stream will begin at ", round_time)
    time.sleep(60+rounding-seconds - (0.000001*microseconds))

now_time = datetime.datetime.now()
print("Current time is ", now_time)

while i < nloops:
    for j in range(0, len(marker_time_series) - 1):
        # publish to stream
        outlet.push_sample(marker_time_series[j])

        if j != len(marker_time_stamps):
            time.sleep((marker_time_stamps[j+1] - marker_time_stamps[j]))
    i += 1
    
# delete the outlet    
print("Deleting marker stream...")
outlet.__del__()
print("Done.")