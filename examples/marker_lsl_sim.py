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
import sys
import time
import datetime
import os

from pylsl import StreamInfo, StreamOutlet

from bci_essentials.sources.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger()
logger.debug("Running %s", __file__)

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "p300_example.xdf")

# check whether to start now, or at the next even minute to sync with other programs
start_now = False
try:
    arg1 = sys.argv[1]
    if arg1 == "now" or arg1 == "-n":
        logger.info("Starting stream immediately")
        start_now = True
except Exception:
    start_now = False

# add the ability to loop n times, default is 1
nloops = 1
try:
    nloops = int(sys.argv[2])
    logger.info("Repeating for %s loops", nloops)

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

i = 0
info = StreamInfo(
    "MockMarker", "LSL_Marker_Strings", 1, fs_marker, "string", "mockmark1"
)
outlet = StreamOutlet(info)

if start_now is False:
    # publish to stream at the next rounded minute
    now_time = datetime.datetime.now()
    logger.info("Current time is %s", now_time)
    seconds = (now_time - now_time.min).seconds
    microseconds = now_time.microsecond
    # // is a floor division, not a comment on following line:
    rounding = (seconds + 60 / 2) // 60 * 60
    round_time = now_time + datetime.timedelta(
        0, 60 + rounding - seconds, -microseconds
    )
    logger.info("microseconds: %s", microseconds)
    logger.info("Stream will begin at %s", round_time)
    time.sleep(60 + rounding - seconds - (0.000001 * microseconds))

now_time = datetime.datetime.now()
logger.info("Current time is %s", now_time)

while i < nloops:
    for j in range(0, len(marker_data) - 1):
        # publish to stream
        outlet.push_sample(marker_data[j])

        if j != len(marker_timestamps):
            time.sleep((marker_timestamps[j + 1] - marker_timestamps[j]))
    i += 1

# delete the outlet
logger.info("Deleting marker stream...")
outlet.__del__()
logger.info("Done.")
