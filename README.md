# bci-essentials-python
This reposity contains python modules and scripts for the processing of EEG-based BCI. 
These modules are specifically designed to be equivalent whether run offline or online.


## Related packages
### bci-essentials-unity
The front end for this package can be found in [bci-essentials-unity](https://www.github.com/kirtonBCIlab/bci-essentials-unity)

## Getting Started
It is recommended to use anaconda or miniconda, all conda commands must be run from the anaconda prompt unless anaconda has been added to PATH.
Alternatively, a regular python virtual environment should be sufficient.

1. clone from git
>git clone https://github.com/kirtonBCIlab/bci-essentials-python.git
2. Create a conda environment from the bci_essentials_env.yml an activate it
>conda env create -f env/bci_essentials_env.yml
>conda activate bci_essentials
2. OR use pip to install to a virtual environment
>pip install virtualenv
>virtualenv bci_essentials
>source bci_essentials/bin/activate
>pip install -r env/bci_essentials_requirements.txt
3. Update pyriemann to the latest version from github
>pip install git+https://github.com/pyRiemann/pyRiemann

## Offline processing
Offline processing can be done by running the corresponding offline test script (ie. mi_offline_test.py, p300_offline_test.py, etc.)
Change the filename in the script to point to the data you want to process.
>python examples/mi_offline_test.py


## Online processing
Online processing requires an EEG stream and a marker stream. These can both be simulated using eeg_lsl_sim.py and marker_lsl_sim.py.
Real EEG streams come from a headset connected over LSL. Real marker streams come from the application in the Unity frontend.
Once these streams are running, simply begin the backend processing script ( ie. mi_unity_backend.py, p300_unity_bakend.py, etc.)
It is recommended to save the EEG, marker, and response (created by the backend processing script) streams using 
[Lab Recorder](https://github.com/labstreaminglayer/App-LabRecorder) for later offline processing.
>python examples/mi_unity_backend.py

## Directory
### bci_essentials
The main packge containing modules for BCI processing.
- bci_data.py         -   module for reading online/offline data, windowing, processing, and classifying EEG signals
- classification.py   -   module containing relevant classifiers for BCI_data, classifiers can be extended to meet individual needs
- signal_processing.py-   module containing functions for the processing of EEG_data
- visuals.py          -   module for visualizing EEG data

### env
Environment files.
- bci_online.yml      -   includes all necessary libraries for BCI essentials plus MNE
- bci_essentials.yml  -   includes all necessary libraries for BCI essentials
- bci_essentials.txt  -   includes all necessary libraries for BCI essentials in the pip requirements format

### examples
Example scripts and data.
- data                        -   directory containing example data for P300, MI, and SSVEP
- eeg_lsl_sim.py              -   creates a stream of mock EEG data from an xdf file
- marker_lsl_sim.py           -   creates a stream of mock marker data from an xdf file
- mi_offline_test.py          -   runs offline MI processing on previously collected EEG and marker streams
- mi_unity_backend.py         -   runs online MI processing on live EEG and marker streams
- p300_offline_test.py        -   runs offline P300 processing on previously collected EEG and marker streams
- p300_unity_backend.py       -   runs online P300 processing on live EEG and marker streams
- ssvep_offline_test.py       -   runs offline SSVEP processing on previously collected EEG and marker streams
- ssvep_unity_backend_tf.py   -   runs online SSVEP processing on live EEG and marker streams, does not require training
- ssvep_unity_backend.py      -   runs online SSVEP processing on live EEG and marker streams
- switch_offline_test.py      -   runs offline switch state processing on previously collected EEG and marker streams
- switch_unity_backend.py     -   runs online switch state processing on live EEG and marker streams


