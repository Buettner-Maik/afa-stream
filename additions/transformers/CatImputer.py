from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
import operator
from numpy import nan

class CategoricalRememberingNaNImputer(TransformerMixin):
    """
    a very simple and specific imputer for categorical values 
    and pd.DataFrame objects that replaces nan values with the 
    most frequent values up until this point

    specifically created for use in the osm framework environment to
    enable missing data to be imputed over several batches without
    value occurences

    not recommended for any other use
    """

    def __init__(self, categories:dict):
        """
        provide all categorical values that will occur in a dict containing lists
        of all values
        these categories are the only ones that will be looped through and imputed
        """
        self.occurences = {}
        for key, values in categories.items():
            self.occurences[key] = {}
            for value in values:
                self.occurences[key][value] = 0

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas.DataFrame object")
        
        #update statistics for all X
        for col in self.occurences.keys():
            for value, count in X[col].value_counts().items():
                #print(f"{col} {value} {count}")
                self.occurences[col][value] += count
            #finds the max in item 2 of all tuples dict.items() provides and returns item 1
            maxval = max(self.occurences[col].items(), key=operator.itemgetter(1))[0]

            X[col].fillna(maxval, inplace=True)
        return X