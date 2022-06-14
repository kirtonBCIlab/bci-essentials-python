# p300_python
Python backend for processing P300 ERP data. Can be used online or offline.

Name will be changing to BCI Essentials Python Backend or similar

#

## Getting Started
This package requires 


1. Make sure that you have git, python, and either anaconda or miniconda installed, the following commands should be entered in the anaconda prompt or in the command line/ terminal (note: to use conda commands directly from cmd you will need to add anaconda to path)
2. clone from git
>git clone https://github.com/ekinney-lang/p300_python.git
3. Create a conda environment from the bci_essentials.yml
>conda env create -f env/bci_online.yml
4. Activate the conda environment bci_essentials
>conda activate bci_essentials
5. Update pyriemann to the latest version from github
>pip install git+https://github.com/pyRiemann/pyRiemann

6. Run erp_offline_test.py to check that the packages are installed (CHANGE THIS TO BE A TEST OF ERP, MI, SSVEP, AND SWITCH)
>python erp_offline_test.py

## Running the Backend
1. Start the game in Unity
2. Start the required backend within your conda environment (ie. p300_unity_backend, ssvep_unity_backend, or mi_unity_backend)
>python p300_unity_backend.py
3. When the Unity application is done, use ctrl^c to quit the backend


## Modules
Modules are found in src/

### bci_data
Handles the data classes for different BCI paradigms. The two main classes are for ERPs (ie. P300) and continuous signals (ie. MI, SSVEP, Switch, etc.) This handles the online and offline reading of LSL streams, stream buffering, and windowing of BCI data. 

### bci_visuals
Contains functions for plotting data from the format provided by bci_data.

### classification_tools
Contains classifier classes for P300, MI, SSVEP, and Switch paradigms. Classifiers have methods to tune settings without changing any code here. Classes can be extended to more complex classifiers.

### signal_processing_tools
Contains functions for preprocessing EEG data.

## Scripts
Scripts meant to be run by the user and can be adapted to suit the needs of the user. Scripts access classes and functions from the above modules.

### p300_offline_test
The P300 offline test injests a given file, processes the signal into windows, builds classifier, tests classifier accuracy.

### erp_online_test
Uses the mock data streams to simulate online classification, primarily to make sure that data injestion can handle data in chunks.

### eeg_lsl_sim
A program for beginning a mock EEG data stream based on a given file from data, start is delayed to the next minute to sync with marker_lsl_sim.py 

### marker_lsl_sim
A program for beginning a mock marker data stream based on a given file from data, start is delayed to the next minute to sync with eeg_lsl_sim.py

### data
Save location for previously collected data.

data/BI - two auditory P300 examples and one visual, pretty mediocre data quality
data/EKL/P300_Diff_Trials - visual P300 examples, better data quality

### env
Environment files. 
bci_online.yml      -   includes all necessary libraries for BCI essentials plus MNE
bci_essentials.yml  -   includes all necessary libraries for BCI essentials
bci_essentials.txt  -   includes all necessary libraries for BCI essentials in the pip requirements format



