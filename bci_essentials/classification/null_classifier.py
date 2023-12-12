"""**Null classifier**

No model is used.
No fitting is done.
The classifier always predicts 0.

"""

# Stock libraries

# Import bci_essentials modules and methods
from ..classification.generic_classifier import GenericClassifier
from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


class NullClassifier(GenericClassifier):
    """NullClassifier class (*inherits from GenericClassifier*)."""

    def fit(self):
        """Fit the classifier to the data.

        No fitting is done.

        Returns
        -------
        `None`

        """
        logger.warning("This is a null classifier, there is no fitting")

    def predict(self, X):
        """Predict the class of the data.

        Parameters
        ----------
        X : numpy.ndarray
            Description of parameter `X`.
            If array, state size and type. E.g.
            3D array containing data with `float` type.

            shape = (`1st_dimension`,`2nd_dimension`,`3rd_dimension`)

        Returns
        -------
        `0`

        """
        logger.warning("This is a null classifier, there is no return value")
        logger.warning("Returning 0")
        return 0
