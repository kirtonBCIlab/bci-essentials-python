import signal
import time
import numpy as np

from pylsl import StreamInfo, StreamOutlet


# Signal handler to stop simulator
def stop_simulator(signum, frame):
    global terminate
    terminate = True


# Install a signal handler to quit cleanly when ^C pressed
terminate = False
signal.signal(signal.SIGINT, stop_simulator)


# This is what comes from an Emotiv EPOC+ headset LSL streamed via EmotivPro
# Using a lower rate to be nice to network, the sim just sends dummy data so this shouldn't matter
fsample = 128.0
psample = 1.0 / fsample
channel_names = [
    "Timestamp",
    "Counter",
    "Interpolate",
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
    "HardwareMarker",
    "Markers",
]
n_channels = len(channel_names)

# Create the outlet StreamInfo with extended data.  Set up stream to look like an Emotiv headset.
# https://github.com/sccn/xdf/wiki/EEG-Meta-Data
stream_info = StreamInfo("MindTV Headset Sim", "EEG", n_channels, fsample, "float32", "MindTVEEG")
channel_info = stream_info.desc().append_child("channels")
for label in channel_names:
    ch = channel_info.append_child("channel")
    ch.append_child_value("label", label)
    ch.append_child_value("unit", "microvolts")
    ch.append_child_value("type", "EEG")

outlet = StreamOutlet(stream_info)

# Print the complete XML description of the stream_info
print(stream_info.as_xml())

# Spew out random samples
while True:
    sample = np.random.uniform(0.0, 1.0, n_channels)
    outlet.push_sample(sample)

    time.sleep(psample)
    if terminate:
        print("Stopping headset simulator")
        break
