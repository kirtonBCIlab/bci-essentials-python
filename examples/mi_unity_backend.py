from bci_essentials.io.lsl_sources import LslEegSource, LslMarkerSource
from bci_essentials.io.lsl_messenger import LslMessenger
from bci_essentials.bci_controller import BciController
from bci_essentials.paradigm.mi_paradigm import MiParadigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.mi_classifier import MiClassifier

# create LSL sources, these will block until the outlets are present
eeg_source = LslEegSource()
marker_source = LslMarkerSource()
messenger = LslMessenger()

paradigm = MiParadigm(live_update=True, iterative_training=True)
data_tank = DataTank()

# Select a classifier
classifier = MiClassifier()  # you can add a subset here

# Set settings
classifier.set_mi_classifier_settings(n_splits=3, type="TS", random_seed=35)

# Define channel selection settings
initial_subset = []
classifier.setup_channel_selection(
    method="SFS",
    metric="accuracy",
    iterative_selection=False,
    initial_channels=initial_subset,  # wrapper setup
    max_time=100,
    min_channels=0,
    max_channels=4,
    performance_delta=-1,  # stopping criterion
    n_jobs=-1,
    record_performance=True,
)

# Define the MI data object
mi_data = BciController(
    classifier, eeg_source, marker_source, paradigm, data_tank, messenger
)

# Run
mi_data.setup(online=True)
mi_data.run()
