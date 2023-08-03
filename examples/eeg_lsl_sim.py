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
from bci_essentials.bci_data import EEG_data

# Add parent directory to path to access bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

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
    food
    nloops = int(sys.argv[2])
    print("repeating for ", nloops, " loops")

except Exception:
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

eeg_keep_ind = [(eeg_time_stamps > time_start) & (eeg_time_stamps < time_stop)]
eeg_time_stamps = eeg_time_stamps[tuple(eeg_keep_ind)]
eeg_time_series = eeg_time_series[tuple(eeg_keep_ind)]


# estimate sampling rates
fs_marker = round(len(marker_time_stamps) / (time_stop - time_start))
fs_eeg = round(len(eeg_time_stamps) / (time_stop - time_start))

# create the eeg stream
info = StreamInfo("MockEEG", "EEG", 8, fs_eeg, "float32", "mockeeg1")

# add channel data
channels = info.desc().append_child("channels")
for c in eeg_stream.channel_labels:
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
    for j in range(0, len(eeg_time_stamps) - 1):
        # publish to stream
        eeg_sample = eeg_time_series[j][:]
        outlet.push_sample(eeg_sample)
        if j != len(eeg_time_stamps):
            time.sleep(eeg_time_stamps[j + 1] - eeg_time_stamps[j])
    i += 1

# delete the outlet
print("Deleting EEG stream")
outlet.__del__()
print("Done.")
