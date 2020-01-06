from osm.data_streams.budget_manager.abstract_budget_manager import AbstractBudgetManager

class SimpleBudgetManager(AbstractBudgetManager):
    """
    returns True whenever another acquisition would not exceed the budget
    """

    def _acquire(self, value, cost = 1):
        """
        returns True if the acquisition would not exceed the budget
        """
        return (self.budget_spent + cost) / self.budget <= self.budget_threshold

    def get_name(self):
        return "simple_budget_manager"
