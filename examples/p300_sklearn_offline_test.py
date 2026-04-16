"""
Test the ErpSklearnClassifier offline using data from an existing stream.

By default, ErpSklearnClassifier uses
    XdawnCovariances(estimator="oas", xdawn_estimator="oas", nfilter=3)
        -> TangentSpace
        -> LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")

This script also demonstrates passing a custom sklearn-style pipeline via
the `clf` argument to `set_p300_clf_settings`.

"""

import os

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.bci_controller import BciController
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.paradigm.p300_paradigm import P300Paradigm
from bci_essentials.classification.erp_sklearn_classifier import (
    ErpSklearnClassifier,
)

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "p300_example.xdf")


def run(clf=None, label=""):
    """Run the offline P300 pipeline with the given sklearn classifier."""
    print(f"\n=== Running ErpSklearnClassifier ({label}) ===")

    eeg_source = XdfEegSource(filename)
    marker_source = XdfMarkerSource(filename)
    paradigm = P300Paradigm()
    data_tank = DataTank()

    classifier = ErpSklearnClassifier()

    # `clf=None` triggers the default XdawnCov(oas, oas, 3) -> TS -> LDA pipeline
    classifier.set_p300_clf_settings(
        clf=clf,
        n_splits=5,
        lico_expansion_factor=4,
        oversample_ratio=0,
        undersample_ratio=0,
        random_seed=35,
        remove_flats=True,
    )

    test_erp = BciController(
        classifier, eeg_source, marker_source, paradigm, data_tank
    )

    test_erp.setup(online=False)
    test_erp.run()

    print(f"[{label}] offline accuracy  = {classifier.offline_accuracy}")
    print(f"[{label}] offline precision = {classifier.offline_precision}")
    print(f"[{label}] offline recall    = {classifier.offline_recall}")


# 1. Default classifier (XdawnCov + TS + LDA)
run(clf=None, label="default LDA")

# 2. Custom sklearn pipeline (XdawnCov + TS + LogisticRegression)
custom_clf = make_pipeline(
    XdawnCovariances(estimator="oas", xdawn_estimator="oas", nfilter=3),
    TangentSpace(metric="riemann"),
    LogisticRegression(max_iter=1000),
)
run(clf=custom_clf, label="custom LR")
