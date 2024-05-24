from bci_essentials.io.lsl_sources import LslEegSource, LslMarkerSource
from bci_essentials.io.lsl_messenger import LslMessenger
from bci_essentials.eeg_data import BciController
from bci_essentials.paradigm.mi_paradigm import MiParadigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.switch_mdm_classifier import SwitchMdmClassifier

# create LSL sources, these will block until the outlets are present
eeg_source = LslEegSource()
marker_source = LslMarkerSource()
messenger = LslMessenger()

paradigm = MiParadigm(live_update=True, iterative_training=True)
data_tank = DataTank()

# LETS TRY IT OUT WITH A WHOLE NEW SWITCH CLASSIFIER
classifier = SwitchMdmClassifier()
classifier.set_switch_classifier_settings(n_splits=3, rebuild=True, random_seed=35)

# Define the SWITCH data object
switch_data = BciController(
    classifier, eeg_source, marker_source, messenger, paradigm, data_tank
)

# Run
switch_data.setup(online=True, training=True)
switch_data.run()
