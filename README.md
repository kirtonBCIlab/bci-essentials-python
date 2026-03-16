# bci-essentials-python
This repository contains python modules and scripts for the processing of EEG-based BCI. 
These modules are specifically designed to be equivalent whether run offline or online.


## Related packages
The front end for this package can be found in [bci-essentials-unity](https://www.github.com/kirtonBCIlab/bci-essentials-unity)

## Getting Started
- **[Wiki](https://docs.bci.games/bessy-python)** – More detailed installation instructions and tutorials.
- **[API documentation](https://kirtonbcilab.github.io/APIdocs-for-bci-essentials-python)**

### Installation
BCI Essentials requires Python 3.9 or later.  To install for Windows, MacOS or Linux:

```
pip install bci-essentials
```

On some systems, it may be necessary to install [liblsl](https://github.com/sccn/liblsl).  Alternatively, use the Conda [environment](https://github.com/kirtonBCIlab/bci-essentials-python/blob/main/environment.yml) to set up dependencies that are not provided by pip:

```
conda env create -f ./environment.yml
conda activate bci
```

## Offline processing
Offline processing can be done by running the corresponding offline test script (ie. mi_offline_test.py, p300_offline_test.py, etc.)
Change the filename in the script to point to the data you want to process.
```
python examples/mi_offline_test.py
```

## Online processing
Online processing requires an EEG stream and a marker stream. These can both be simulated using eeg_lsl_sim.py and marker_lsl_sim.py.
Real EEG streams come from a headset connected over LSL. Real marker streams come from the application in the Unity frontend.
Once these streams are running, simply begin the backend processing script ( ie. mi_unity_backend.py, p300_unity_backend.py, etc.)
It is recommended to save the EEG, marker, and response (created by the backend processing script) streams using
[Lab Recorder](https://github.com/labstreaminglayer/App-LabRecorder) for later offline processing.
```
python examples/mi_unity_backend.py
```

### Serial port triggers
As an alternative to LSL marker streams, markers can be delivered via a hardware trigger box
(e.g. MMBT-S) connected to the EEG amplifier's stim channel. Unity's `SerialTriggerWriter`
sends byte values over a serial port, the trigger box forwards them to the amplifier, and
`LslStimMarkerSource` reads them from the EEG stream's stim channel.

```python
from bci_essentials.io.lsl_sources import LslEegSource
from bci_essentials.io.lsl_stim_marker_source import LslStimMarkerSource
from bci_essentials.triggers import make_p300_trigger_map

eeg_source = LslEegSource()
marker_source = LslStimMarkerSource(
    stim_channel_name="TRG",
    trigger_map=make_p300_trigger_map(n_options=6),
)
```

Trigger maps for MI and SSVEP paradigms are also available via `make_mi_trigger_map` and
`make_ssvep_trigger_map` in `bci_essentials.triggers`.

## Development

```
make install   # install package
make dev-install  # install package in editable mode (pip install -e .) and development dependencies (black, flake8)
make test      # run tests (python -m unittest)
make lint      # run black --check and flake8 (same as CI)
```

## Directory
### bci_essentials
The main package containing modules for BCI processing.
- bci_controller.py         -   module for reading online/offline data, windowing, processing, and classifying EEG signals
- classification.py   -   module containing relevant classifiers for bci_controller, classifiers can be extended to meet individual needs
- signal_processing.py-   module containing functions for the processing of bci_controller
- visuals.py          -   module for visualizing EEG data

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


