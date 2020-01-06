from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from osm.data_streams.abstract_base_class import AbstractBaseClass
import osm.data_streams.constants as const

class AbstractBudgetManager(AbstractBaseClass):
    def __init__(self, budget_threshold, initial_budget = 0, default_budget_gain = 1):
        """
        decides whether to acquire a feature or not given a budget and quality value
        :param budget_threshold: The relative amount of budget that can be spent
        :param initial_budget: The starting budget available
        :param default_budget_gain: The default budget added when add_budget is called without params
        """
        super().__init__()

        if not isinstance(budget_threshold, np.float):
            raise ValueError("The budget_threshold should be a float between [0,1]")

        if budget_threshold < 0 or budget_threshold > 1:
            raise ValueError("The budget_threshold should be a float between [0,1]")

        self.budget_threshold = budget_threshold
        self.budget = initial_budget
        self.budget_spent = 0
        self.default_budget_gain = default_budget_gain
        #statistic relevant variables
        self.called = 0
        self.queried = 0

    def used_budget(self):
        """
        the relative budget used
        """
        #ignore 0 / 0 as an error
        if self.budget == 0:
            return np.nan
        return self.budget_spent / self.budget

    def acquire(self, value, cost = 1):
        """
        whether to acquire something given the current budget
        calls the private function in actual implementations
        """
        self.called += 1
        if self._acquire(value, cost):
            self.queried += 1
            self.budget_spent += cost
            return True
        return False

    @abstractmethod
    def _acquire(self, value, cost = 1):
        """
        whether to acquire something given the current budget
        implement self.budget_spent / self.called
        """
        pass

    def add_budget(self, value = None):
        """
        raises the absolute budget by value
        leave None for default_budget_gain
        """
        if value is None:
            value = self.default_budget_gain
        self.budget += value

    def get_stats(self, index=0):
        pd_index = pd.MultiIndex.from_product([[const.budget_manager_stats], [const.queried, const.answered, const.budget_used]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        stats.loc[index, (const.budget_manager_stats, const.queried)] = self.called
        stats.loc[index, (const.budget_manager_stats, const.answered)] = self.queried
        stats.loc[index, (const.budget_manager_stats, const.budget)] = self.budget
        stats.loc[index, (const.budget_manager_stats, const.budget_spent)] = self.budget_spent
        stats.loc[index, (const.budget_manager_stats, const.budget_used)] = self.used_budget()

        return stats