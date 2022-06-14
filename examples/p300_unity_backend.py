
import sys

from pylsl import StreamInlet, resolve_stream, resolve_byprop

from bci_essentials.bci_data import *

# Initialize the ERP
# should try to automate the reading of some of this stuff from the file header
#test_erp = ERP_data(user_id='0001', nchannels=13, channel_locations=['Fz','Cz','P3','Pz','P4','PO3','POz','PO4','PO7','O1','Oz','O2','PO8'], fsample=256)
test_erp = ERP_data()
#test_erp.edit_settings(user_id='0002', nchannels=8, channel_labels=['Cz','P3','Pz','P4','POz','O1','Oz','O2'], fsample=256)

# Set classifier settings ()
test_erp.classifier = erp_rg_classifier()
test_erp.classifier.set_p300_clf_settings(n_splits=5, lico_expansion_factor=1, oversample_ratio=1, undersample_ratio=0)

# # Load a template
# test_template = Template.load_template(file_path = "./templates", file_name = "ekl_base")

# Connect the streams
test_erp.stream_online_eeg_data()

# Run

test_erp.main(online=True, training=True, pp_low=0.1, pp_high=15, pp_order=5, plot_erp=False, window_start=0.0, window_end=0.6)
#test_erp.main(online=True, training=True, pp_type="none", plot_erp=False)