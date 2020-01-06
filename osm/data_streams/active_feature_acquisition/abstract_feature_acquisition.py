from abc import ABC, abstractmethod

import numpy as np
import math
import pandas as pd
from datetime import datetime

from osm.data_streams.abstract_base_class import AbstractBaseClass
from osm.data_streams.budget_manager.abstract_budget_manager import AbstractBudgetManager
from osm.data_streams.budget_manager.no_budget_manager import NoBudgetManager
import osm.data_streams.constants as const

class AbstractActiveFeatureAcquisitionStrategy(AbstractBaseClass):
    def __init__(self, target_col_name, budget_manager, put_back=False, acquisition_costs={}, debug=True):
        """
        Abstract class to implement an active feature acquisition strategy
        :param budget_manager: An instance of type AbstractBudgetManager
        :param acquisition_costs: A dictionary with the names of columns as key and acquisition_cost as value
        if a field is left empty, handle like set to 1
        :param target_col_name: The name of the target column
        :param put_back: Whether instances may be evaluated multiple times for acquisition
        :param debug: If True prints debug messages to console
        """
        super().__init__()

        #if not isinstance(budget, np.float):
        #    raise ValueError("The budget should be a float between [0,1]")

        #if budget < 0 or budget > 1:
        #    raise ValueError("The budget should be a float between [0,1]")

        if budget_manager is None or not isinstance(budget_manager, AbstractBudgetManager):
            raise ValueError("The budget_manager must be of type AbstractBudgetManager")

        self.put_back = put_back
        self.queried = {}
        self.target_col_name = target_col_name
        self.acquisition_costs = acquisition_costs
        self.budget_manager = budget_manager
        self.debug = debug

    def _initialize_queried(self, columns):
        """
        Call to initialize self.queried for use as statistic
        :param columns: The specified columns as keys for self.queried
        """
        for col in columns:
            self.queried[col] = 0

    def get_columns_with_nan(self, data, columns=None):
        """
        returns a list of column names that contain missing data
        :param data: the data to check for missing entries
        :param columns: specify which columns to check
        """
        if columns is None:
            return data.columns[data.isna().any()].tolist()
        else:
            return data.columns[data[columns].isna().any()].tolist()

    def isnan(self, value):
        """
        checks whether a value is nan
        """
        if not isinstance(value, float):
            return False
        return math.isnan(value)
			
    def has_nan_value(self, data, columns=None):
        """
        returns true if data has nan values
        :param data: the data to check for missing entries
        :param columns: specify which columns to check
        """
        if columns is None:
            return data.isna().any().any()
        else:
            return data[columns].isna().any().any()

    def label_known(self, inst):
        """
        returns whether the label of inst is known
        """
        return self.target_col_name in inst and not math.isnan(inst[self.target_col_name])

    def get_feature(self, inst, column):
        """
        returns the value for the requested feature
        currently just returns a column with same name + "_org"
        TODO: export get_feature as separate module or make choice to use separate data
        """
        return inst[column + '_org']

    @abstractmethod
    def get_data(self, data):
        """
        gets the data with additionally requested features
        """
        pass
        
    def get_stats(self, index=0):
        pd_index = pd.MultiIndex.from_product([[const.active_feature_acquisition_queries], [const.queried, const.answered, const.budget_used]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        stats.loc[index, (const.active_feature_acquisition_queries, const.queried)] = self.budget_manager.called
        stats.loc[index, (const.active_feature_acquisition_queries, const.answered)] = self.budget_manager.queried
        stats.loc[index, (const.active_feature_acquisition_queries, const.budget_used)] = self.budget_manager.used_budget()

        return stats

        #if self.debug:
            #print(str.format("{0}: Index: {1}\tQueried: {2}\tAnswered: {3}\tCost: {4}",
                  #str(datetime.now()),
                  #index,
                  #self.oracle.get_total_queried(),
                  #self.oracle.get_total_answered(),
                  #self.oracle.get_cost()))