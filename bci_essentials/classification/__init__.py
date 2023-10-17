"""
Classifiers
-----------
Compendium of classifiers for BCI applications.

**Excluded from `flake8` rules to avoid unused import errors**
"""

from .generic_classifier import Generic_classifier
from .erp_rg_classifier import ERP_rg_classifier
from .mi_classifier import MI_classifier
from .ssvep_riemannian_mdm_classifier import SSVEP_riemannian_mdm_classifier
from .ssvep_basic_tf_classifier import SSVEP_basic_tf_classifier
from .null_classifier import Null_classifier
from .switch_mdm_classifier import Switch_mdm_classifier
from .switch_deep_classifier import Switch_deep_classifier
