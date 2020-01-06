import pandas as pd
import numpy as np
import math

from osm.data_streams.active_feature_acquisition.abstract_feature_acquisition import AbstractActiveFeatureAcquisitionStrategy

class RandomFeatureAcquisition(AbstractActiveFeatureAcquisitionStrategy):
    def __init__(self, target_col_name, budget_manager, put_back=False, columns=None, acquisition_costs={}, debug=True):
        """
        An active feature acquisition method that selects features to request at random
        :param columns: If None gets all columns to pick from on first call of get_data
        from provided data
        :param budget_manager: The Budgeting strategy used will only be given None as parameter
        for its acquisition
        """
        super().__init__(target_col_name=target_col_name, 
                        budget_manager=budget_manager, 
                        acquisition_costs=acquisition_costs,
                        put_back=put_back,
                        debug=debug)
        self.columns = columns
        self._initialized = False
    
    def _initialize(self, data):
        """
        Gets the columns the data has for use in the random number generator and
        prepares self.queried
        """
        if self.columns is None:
            self.columns = list(data.columns.values)
            self.columns.remove(self.target_col_name)

        self._initialize_queried(self.columns)
        self._initialized = True

    def get_data(self, data):
        """
        Returns the incoming data with additionally acquired features
        :param data: A pandas.DataFrame with the data to do active feature acquisition on
        """
        if not self._initialized:
            self._initialize(data)

        if hasattr(data, "iterrows"):
            for index, row in data.iterrows():
                #get all nan containing col indices in row
                nan_cols = []
                i = 0
                for col in self.columns:
                    if self.isnan(row[col]):
                        nan_cols.append(i)
                    i += 1
                while len(nan_cols) > 0:
                    self.budget_manager.add_budget()
                    #Randomize which to pick
                    rnd = math.floor(np.random.uniform(high=len(nan_cols)))
                    col = self.columns[nan_cols[rnd]]
                    nan_cols.pop(rnd)

                    #apply budgeting
                    if self.budget_manager.acquire(0.0,
                                                  self.acquisition_costs.get(col, 1)):
                        feature_val = self.get_feature(row, col)
                        data.loc[[index], [col]] = feature_val
                        self.queried[col] += 1
                        
                        #continue with next if no put_back
                        if not self.put_back:
                            break
                    else:
                        break
        else:
            raise ValueError("A pandas.DataFrame expected")

        return data

    def get_name(self):
        return "random_acquisition"
