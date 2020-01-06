from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
from numpy import nan
from scipy.sparse import issparse

class DenseTransformer(TransformerMixin):
    """
    A transformer with the sole purpose to convert an incoming sparse matrix into a dense one
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if issparse(X):
            return X.todense()
        return X
