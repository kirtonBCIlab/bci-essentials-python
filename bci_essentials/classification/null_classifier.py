"""**Null classifier**

No model is used.
No fitting is done.
The classifier always predicts 0.

"""

# Stock libraries
import os
import sys

from bci_essentials.classification import Generic_classifier

# Custom libraries
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


class Null_classifier(Generic_classifier):
    """Null Classifier class (*inherits from Generic_classifier*)."""

    def fit(self, print_fit=True, print_performance=True):
        """Fit the classifier to the data.

        No fitting is done.

        Parameters
        ----------
        print_fit : bool, *optional*
            Description of parameter `print_fit`.
            - Default is `True`.
        print_performance : bool, *optional*
            Description of parameter `print_performance`.
            - Default is `True`.

        Returns
        -------
        `None`

        """
        print("This is a null classifier, there is no fitting")

    def predict(self, X, print_predict):
        """Predict the class of the data.

        Parameters
        ----------
        X : numpy.ndarray
            Description of parameter `X`.
            If array, state size and type. E.g.
            3D array containing data with `float` type.

            shape = (`1st_dimension`,`2nd_dimension`,`3rd_dimension`)
        print_predict : bool
            Description of parameter `print_predict`.

        Returns
        -------
        `0`

        """
        return 0
