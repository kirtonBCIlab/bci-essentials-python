import sys
import time
import pyxdf
import numpy as np
import mne
from mne.datasets import misc
from mne.annotations import Annotations
import matplotlib.pyplot as plt

from bci_essentials.bci_data import *
from bci_essentials.visuals import *



# Select a file
#filename = "C:/Users/brian/Documents/OptimizationStudy/TestData/P300/March29/BI/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-StandardSingle_run-001_eeg.xdf"
#filename = "C:/Users/brian/Documents/OptimizationStudy/TestData/P300/April8/BI/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-StandardSingle_run-001_eeg.xdf"
#filename = "C:/Users/brian/Documents/OptimizationStudy/TestData/P300/April6/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-SingleFlash_run-001_eeg.xdf"
filename = "C:/Users/brian/Documents/OptimizationStudy/TestData/P300/April12/BI/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-StandardSingle_run-004_eeg.xdf"

#multi
#filename = "C:/Users/brian/Documents/OptimizationStudy/TestData/P300/March29/BI/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-StandardMulti_run-001_eeg.xdf"

# If multiple files selected



# Break down the xdf
test_erp = ERP_data()
test_erp.load_offline_eeg_data(filename, format="xdf")


# test_erp.eeg_data[:,:23] = test_erp.eeg_data[:,:23] - ()

# Parse EEG to MNE
info = mne.create_info(test_erp.channel_labels, test_erp.fsample, ["eeg"] * test_erp.nchannels)
raw = mne.io.RawArray(np.transpose(test_erp.eeg_data[:,:23]), info)
raw.filter(l_freq=0.1, h_freq=15)
mne.set_eeg_reference(raw, ref_channels="average")

print(info)
#raw.plot(scalings=dict(eeg=20), duration=1, start=10)


# Transform markers to event format

# P300
# marker array
# onset, duration, value, stim_file
stim_index = 0
stim_table = list()

stim_table = [ [ [] for i in range(4) ] for i in range(1000) ]
stim_array = np.ndarray([1000,3], dtype=int)
erp_stim_array = np.ndarray([1000,3], dtype=int)

t0 = test_erp.eeg_timestamps[0]

for i in range(len(test_erp.marker_timestamps)):
    # if it is a regular marker
    marker_string = test_erp.marker_data[i][0]
    markers = marker_string.split(",")

    #DIVDE BY THOUSAND FOR NEUROSITY
    timestamp = test_erp.marker_timestamps[i] - t0

    if len(markers) >= 5 and markers[0] == 'p300':
        # if length is greater than 5 (ie. multiflash) then we need to create an event for each
        for j in range(len(markers) - 4):
            # onset
            stim_table[stim_index][0] = timestamp
            stim_array[stim_index,0] = int(np.round(timestamp * test_erp.fsample)) 
            erp_stim_array[stim_index,0] = int(np.round(timestamp * test_erp.fsample)) 
            
            # duration (will have to add this in)
            stim_table[stim_index][1] = 0.2
            #stim_array[stim_index,1] = 0.2
            
            stim_array[stim_index,1] = 0
            erp_stim_array[stim_index,1] = 0
            

            # value
            stim_table[stim_index][2] = markers[4+j]

            
            stim_array[stim_index, 2] = markers[4+j]

            # save the value if it is an ERP or 99 otherwise
            if markers[4+j] == markers[3]:
                erp_stim_array[stim_index, 2] = markers[3]

            else:
                erp_stim_array[stim_index, 2] = 99

            # stim file
            stim_table[stim_index][3] = "flash"

            # increment the stim index
            stim_index += 1

#print(stim_table)

events = np.array(stim_array[:stim_index,:])
erp_events = np.array(erp_stim_array[:stim_index,:])


only_erp_events = erp_events[erp_events[:,2] != 99] 

difs = only_erp_events[1:] - only_erp_events[:-1]
otherdifs = events[1:] - events[:-1]



#annotations = Annotations()
# Load the stim table into MNE


# Visualize

#raw.filter(l_freq=1, h_freq=2)
#mne.set_eeg_reference(raw, ref_channels="average")

epochs = mne.Epochs(raw, only_erp_events, tmin=-0.1, tmax=0.6, preload=True, event_repeated='merge', picks=['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'Pz', 'Fp1', 'Fp2', 'T3', 'T5', 'O1', 'O2','F7', 'F8', 'T6', 'T4'])
#epochs = mne.Epochs(raw, only_erp_events, tmin=-0.1, tmax=0.8, preload=True, event_repeated='merge', picks=['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'Pz'])

epochs
fig = epochs.plot(scalings="auto")

#fig11 = mne.viz.plot_epochs(epochs, scalings="auto")

num_objects = 9
averages = [0] * num_objects
for i in range(num_objects):
    averages[i] = epochs[i].average()
    
#nonerp = epochs[99].average()


fig0 = averages[0].plot(spatial_colors=True)
fig0.show()

fig1 = averages[1].plot(spatial_colors=True)
fig1.show()

fig2 = averages[2].plot(spatial_colors=True)
fig2.show()

fig3 = averages[3].plot(spatial_colors=True)
fig3.show()

fig4 = averages[4].plot(spatial_colors=True)
fig4.show()

fig5 = averages[5].plot(spatial_colors=True)
fig6 = averages[6].plot(spatial_colors=True)
fig7 = averages[7].plot(spatial_colors=True)
fig8 = averages[8].plot(spatial_colors=True)

# fig9 = nonerp.plot()
# fig9.show()







# for proj in (False, True):
#     with mne.viz.use_browser_backend('matplotlib'):
#         fig = raw.plot(n_channels=5, proj=proj, scalings=dict(eeg=20))
#     fig.subplots_adjust(top=0.9)  # make room for title
#     ref = 'Average' if proj else 'No'
#     fig.suptitle(f'{ref} reference', size='xx-large', weight='bold')

print("howdy")