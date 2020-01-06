from sklearn.preprocessing import LabelEncoder
import pandas as pd

class PipelineLabelEncoder(LabelEncoder):
    """
    LabelEncoder made compatible for use in pipelines
    """
    def fit(self, X, y):
        return super().fit(self, X)
    
    def fit_transform(self, X, y):
        return super().fit_transform(self, X)
    
    def transform(self, X, y):
        return super().transform(self, X)