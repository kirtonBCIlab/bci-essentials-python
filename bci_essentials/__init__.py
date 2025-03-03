# Import submodules to make them discoverable
from . import classification
from . import data_tank
from . import io
from . import paradigm
from . import utils

# Instantiate a parent logger for the library at default
from .utils.logger import Logger

logger = Logger()
