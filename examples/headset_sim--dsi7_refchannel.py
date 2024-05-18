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
fsample = 300.0
psample = 1.0 / fsample
channel_names = [
    "S2",
    "F4",
    "C4",
    "S3",
    "S1",
    "C3",
    "F3"
]
n_channels = len(channel_names)

# Create the outlet StreamInfo with extended data.  Set up stream to look like an Emotiv headset.
# https://github.com/sccn/xdf/wiki/EEG-Meta-Data
stream_info = StreamInfo("DSI7 Headset Sim", "EEG", n_channels, fsample, "float32", "SIMULATED EEG")
channel_info = stream_info.desc().append_child("channels")
print(channel_info)
for label in channel_names:
    ch = channel_info.append_child("channel")
    ch.append_child_value("label", label)
    ch.append_child_value("unit", "microvolts")
    ch.append_child_value("type", "EEG")
# Add reference channel information
reference = stream_info.desc().append_child("reference")
reference.append_child_value("label", "A1A2")

outlet = StreamOutlet(stream_info)

# Print the complete XML description of the stream_info
print(stream_info.as_xml())

# Spew out random samples
while True:
    sample = np.random.uniform(0.0, 1.0, n_channels)
    outlet.push_sample(sample)
    # print(sample)

    time.sleep(psample)
    if terminate:
        print("Stopping headset simulator")
        break
