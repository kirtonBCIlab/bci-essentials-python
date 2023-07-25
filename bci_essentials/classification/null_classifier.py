# Stock libraries
import os
import sys

# Custom libraries
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))
from classification.generic_classifier import Generic_classifier

class Null_classifier(Generic_classifier):
    def fit(self, print_fit=True, print_performance=True):
        print("This is a null classifier, there is no fitting")

    def predict(self, X, print_predict):
        return 0