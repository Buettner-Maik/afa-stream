from blist import sortedlist
import numpy as np
import math

from osm.data_streams.budget_manager.abstract_budget_manager import AbstractBudgetManager

#minimalist value to be added to values to avoid same value sorting
epsilon = 1 / (2 ** 20)

class IncrementalPercentileFilter(AbstractBudgetManager):
    def __init__(self, budget_threshold, window_size):
        """
        handles inline budgeting of streams by sorting merit values and deciding acquisition 
        on their rank in the list
        ignores costs

        usable only for uniform acquisition costs
        :param budget_threshold:The relative amount of budget that can be spent
        in terms of queried / called; ignores budget and budget_spent
        :param window_size:The size of the list in which merit values are ranked
        """
        super().__init__(budget_threshold=budget_threshold)

        self.counter = 0
        self.window_size = window_size
        self.values_list = [None] * window_size
        self.values = sortedlist()

    def _acquire(self, value, cost = 1):
        """
        adds a value to its list and returns whether to acquire the value or not
        :param value: a float value akin to a merit or quality
        """
        if not isinstance(value, np.float):
            raise ValueError("value must be a float")

        self.counter = self.counter + 1
        #randomize to ensure no equal values
        value += np.random.uniform() * epsilon
        i = (self.counter-1) % self.window_size

        #replace oldest value if window size reached
        if self.counter > self.window_size:
            oldest_val = self.values_list[i]
            self.values.remove(oldest_val)

        self.values_list[i] = value
        self.values.add(value)

        return self.values[math.floor(min(self.window_size, self.counter) * (1 - self.budget_threshold))] <= value

    def get_name(self):
        return "incremental_percentile_filter"
