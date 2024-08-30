from bci_essentials.io.lsl_sources import LslEegSource, LslMarkerSource
from bci_essentials.io.lsl_messenger import LslMessenger
from bci_essentials.bci_controller import BciController
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.paradigm.ssvep_paradigm import SsvepParadigm
from bci_essentials.classification.ssvep_basic_tf_classifier import (
    SsvepBasicTrainFreeClassifier,
)

# create LSL sources, these will block until the outlets are present
eeg_source = LslEegSource()
marker_source = LslMarkerSource()
messenger = LslMessenger()

paradigm = SsvepParadigm()
data_tank = DataTank()

# Define the classifier
classifier = SsvepBasicTrainFreeClassifier()

# Initialize the EEG Data
test_ssvep = BciController(
    classifier, eeg_source, marker_source, paradigm, data_tank, messenger
)

# set train complete to true so that predictions will be allowed
test_ssvep.train_complete = True

classifier.target_freqs = [7.857143, 9.705882, 12.69231, 15, 18.33333, 22]

# Run
test_ssvep.setup(online=True)
test_ssvep.run()
