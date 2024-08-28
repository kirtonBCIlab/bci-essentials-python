from bci_essentials.io.lsl_sources import LslEegSource, LslMarkerSource
from bci_essentials.io.lsl_messenger import LslMessenger
from bci_essentials.bci_controller import BciController
from bci_essentials.paradigm.ssvep_paradigm import SsvepParadigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.ssvep_riemannian_mdm_classifier import (
    SsvepRiemannianMdmClassifier,
)

# create LSL sources, these will block until the outlets are present
eeg_source = LslEegSource()
marker_source = LslMarkerSource()
messenger = LslMessenger()

paradigm = SsvepParadigm()
data_tank = DataTank()

# Define the classifier
classifier = SsvepRiemannianMdmClassifier()
classifier.set_ssvep_settings(
    n_splits=5, random_seed=42, n_harmonics=3, f_width=0.5, covariance_estimator="oas"
)

# Initialize the data class
test_ssvep = BciController(
    classifier, eeg_source, marker_source, paradigm, data_tank, messenger
)

classifier.target_freqs = [
    7.857143,
    9.705882,
    12.69231,
    15,
    18.33333,
    22
]

# # Channel Selection
# initial_subset=[]
# test_ssvep.classifier.setup_channel_selection(method = "SBFS", metric="accuracy", initial_channels = initial_subset,    # wrapper setup
#                                 max_time= 999, min_channels=2, max_channels=14, performance_delta=0,      # stopping criterion
#                                 n_jobs=-1)

test_ssvep.setup(
    online=True,
)
test_ssvep.run()
