#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG LSL sim simulates offline EEG data as online data

Additional Arguments:
now (-n)            -   start the stream immediately
num_loops (int)     -   number of times to repeat the stream

ex.
>python eeg_lsl_sim.py now 8
Immediately begins the simulated stream, repeats 8 times

Created on Wed Apr 21 10:26:44 2021

@author: brianirvine
"""
import os
import sys
import time
import datetime

from pylsl import StreamInfo, StreamOutlet

# Import local bci_essentials
from bci_essentials.sources.xdf_sources import XdfEegSource, XdfMarkerSource

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "p300_example.xdf")

# Check whether to start now, or at the next even minute to sync with other programs
start_now = False
try:
    arg1 = sys.argv[1]
    if arg1 == "now" or arg1 == "-n":
        print("starting stream immediately")
        start_now = True
except Exception:
    start_now = False

# add the ability to loop n times, default is 1
nloops = 1
try:
    nloops = int(sys.argv[2])
    print("repeating for ", nloops, " loops")

except Exception:
    nloops = 1

# Load the example EEG / marker streams
marker_source = XdfMarkerSource(filename)
eeg_source = XdfEegSource(filename)

# Get the data from that stream
marker_data, marker_timestamps = marker_source.get_markers()
eeg_data, eeg_timestamps = eeg_source.get_samples()

# find the time range of the marker stream and delete EEG data out of this range
time_start = min(marker_timestamps)
time_stop = max(marker_timestamps)

eeg_keep_ind = [(eeg_timestamps > time_start) & (eeg_timestamps < time_stop)]
eeg_timestamps = eeg_timestamps[tuple(eeg_keep_ind)]
eeg_data = eeg_data[tuple(eeg_keep_ind)]

# estimate sampling rates
fs_marker = round(len(marker_timestamps) / (time_stop - time_start))
fs_eeg = round(len(eeg_timestamps) / (time_stop - time_start))

# create the eeg stream
info = StreamInfo("MockEEG", "EEG", 8, fs_eeg, "float32", "mockeeg1")

# add channel data
channels = info.desc().append_child("channels")
for c in eeg_source.channel_labels:
    channels.append_child("channel").append_child_value("name", c).append_child_value(
        "unit", "microvolts"
    ).append_child_value("type", "EEG")

# create the EEG stream
outlet = StreamOutlet(info)

if start_now is False:
    # publish to stream at the next rounded minute
    now_time = datetime.datetime.now()
    print("Current time is ", now_time)
    seconds = (now_time - now_time.min).seconds
    microseconds = now_time.microsecond
    # // is a floor division, not a comment on following line:
    rounding = (seconds + 60 / 2) // 60 * 60
    round_time = now_time + datetime.timedelta(
        0, 60 + rounding - seconds, -microseconds
    )
    print(microseconds)
    print("Stream will begin at ", round_time)
    time.sleep(60 + rounding - seconds - (0.000001 * microseconds))

now_time = datetime.datetime.now()
print("Current time is ", now_time)

i = 0
while i < nloops:
    for j in range(0, len(eeg_timestamps) - 1):
        # publish to stream
        eeg_sample = eeg_data[j][:]
        outlet.push_sample(eeg_sample)
        if j != len(eeg_timestamps):
            time.sleep(eeg_timestamps[j + 1] - eeg_timestamps[j])
    i += 1

# delete the outlet
print("Deleting EEG stream")
outlet.__del__()
print("Done.")
