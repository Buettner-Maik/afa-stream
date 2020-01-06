from osm.data_streams.budget_manager.abstract_budget_manager import AbstractBudgetManager

class NoBudgetManager(AbstractBudgetManager):
    def __init__(self):
        """
        Always returns true on acquire; use if no budget 
        """
        super().__init__(budget_threshold=1.0)    
    
    def _acquire(self, value, cost = 1):
        """
        returns True
        """
        return True

    def get_name(self):
        return "no_budget_manager"
