"""
BCI Essentials Classification Submodule

This package contains various classifier implementations for different BCI paradigms:
- Generic classifiers (base implementations)
- ERP (Event-Related Potential) classifiers
- MI (Motor Imagery) classifiers
- SSVEP (Steady-State Visual Evoked Potential) classifiers
- Utility classifiers
"""

# Import all classifier modules to make them discoverable
from . import generic_classifier
from . import erp_rg_classifier
from . import erp_single_channel_classifier
from . import mi_classifier
from . import null_classifier
from . import ssvep_basic_tf_classifier
from . import ssvep_riemannian_mdm_classifier
from . import switch_mdm_classifier