"""This module contains a custom transformer that reduces the dimensionality of the input data by removing all but the first channel to turn it into a 2D array."""

from sklearn.base import BaseEstimator, TransformerMixin


class ReduceToSingleChannel(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # No fitting necessary for this transformer
        return self

    def transform(self, X):
        if X.ndim == 3:
            X = X[:, 0, :]
        return X
