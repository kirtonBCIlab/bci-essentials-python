from bci_essentials.erp_data import ERP_data
from bci_essentials.classification.erp_rg_classifier import ERP_rg_classifier

# Set classifier settings ()
classifier = ERP_rg_classifier()  # you can add a subset here

# Set some settings
classifier.set_p300_clf_settings(
    n_splits=5,
    lico_expansion_factor=1,
    oversample_ratio=0,
    undersample_ratio=0,
    covariance_estimator="oas",
)

# Initialize the ERP
test_erp = ERP_data(classifier)

# Connect the streams
test_erp.stream_online_eeg_data()  # you can also add a subset here

# Run main
test_erp.main(
    online=True,
    training=True,
    pp_low=0.1,
    pp_high=10,
    pp_order=5,
    plot_erp=False,
    window_start=0.0,
    window_end=0.8,
    print_markers=False,
    print_training=False,
    print_fit=False,
    print_performance=True,
    print_predict=False,
)
