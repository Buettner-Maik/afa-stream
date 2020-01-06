from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
from numpy import nan

class MissTransformer(TransformerMixin):
    """
    Creates additional columns for each original column in the DataFrame with its value 
    replaced by nan given the chance in miss_chance
    the original columns are renamed to name + _org
    """

    def __init__(self, miss_chances) -> None:
        super().__init__()
        if not type(miss_chances) is dict:
            raise ValueError("miss_chance has to be a dict")
        self.miss_chances = miss_chances

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            raise ValueError("x must be a pandas.DataFrame object")

        for col, value in self.miss_chances.items():
            print('Insert NaNs in ' + col)
            X[col + '_org'] = X[col]
            colindex = X.columns.get_loc(col)
            rans = np.random.uniform(0.0, 1.0, X.shape[0])
            for i in range(X.shape[0]):
                if rans[i] < value:
                    X.iat[i, colindex] = nan

            #print(col + ' ' + str(X[col].count()))
        
        return X
