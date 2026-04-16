"""
P300 Unity backend using ErpSklearnClassifier.

By default, ErpSklearnClassifier uses
    XdawnCovariances(estimator="oas", xdawn_estimator="oas", nfilter=3)
        -> TangentSpace
        -> LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")

To use a different sklearn-style estimator, build it and pass it via the
`clf` argument to `set_p300_clf_settings`.

"""

from bci_essentials.io.lsl_sources import LslEegSource, LslMarkerSource
from bci_essentials.io.lsl_messenger import LslMessenger
from bci_essentials.bci_controller import BciController
from bci_essentials.paradigm.p300_paradigm import P300Paradigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.erp_sklearn_classifier import (
    ErpSklearnClassifier,
)

# create LSL sources, these will block until the outlets are present
eeg_source = LslEegSource()
marker_source = LslMarkerSource()
messenger = LslMessenger()
paradigm = P300Paradigm()
data_tank = DataTank()

# Set classifier settings ()
classifier = ErpSklearnClassifier()  # you can add a subset here

# `clf=None` triggers the default XdawnCov(oas, oas, 3) -> TS -> LDA pipeline.
# Pass any sklearn-style estimator/pipeline via `clf=` to swap it out.
classifier.set_p300_clf_settings(
    clf=None,
    n_splits=5,
    lico_expansion_factor=1,
    oversample_ratio=0,
    undersample_ratio=0,
)

# Initialize the ERP
test_erp = BciController(
    classifier, eeg_source, marker_source, paradigm, data_tank, messenger
)

# Run main
test_erp.setup(
    online=True,
)
test_erp.run()
