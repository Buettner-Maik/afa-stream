from abc import ABC, abstractmethod
import pandas as pd
from pandas.api.types import is_numeric_dtype
import math
import operator
import copy
import osm.data_streams.constants as const

from osm.data_streams.windows.abstract_window import AbstractWindow

from osm.data_streams.active_feature_acquisition.abstract_feature_acquisition import AbstractActiveFeatureAcquisitionStrategy

class AbstractSMR(AbstractActiveFeatureAcquisitionStrategy):
    #TODO: integrate differing costs
    #TODO: get function to tell if label known
    #TODO: get the acquisition module
    def __init__(self, window, target_col_name, budget_manager, put_back=False, categories=None, acquisition_costs={}, budget_option=[('acq', 1)], debug=True):
        """
        Abstract Active Feature Acquisition strategies that rank entire features accoring to calculated merits
        and chooses features to acquire iteratively
        Requires the exact categories and possible category values beforehand
        Abstract basis for other different merit mechanisms
        Implement get_data(self, data) to be used
        :param window: the framework window containing the initially labeled data
        :param put_back: whether an instance once acquired a feature will enter the decision process of 
        acquisition again
        :param categories: provide categories for all categorical features as dict         
        empty categorical names are interpreted as numerical columns
        alternatively if left None make initally labeled data in window represent 
        all categorical features at least once 
        :param budget_option: specify a list of tuples consisting of points at which budget is distributed and budget gain function
        accepted strings as position: 'once', 'batch', 'inst', 'acq'
        """
        if not categories == None and not isinstance(categories, dict):
            raise ValueError("categories is not a dictionary nor left none")
        if not isinstance(window, AbstractWindow):
            raise ValueError("The window containing the ild must be the same instance of AbstractWindow as the framework uses")
        super().__init__(target_col_name=target_col_name,
                            budget_manager=budget_manager,
                            acquisition_costs=acquisition_costs,
                            put_back=put_back,
                            debug=debug)
        self.merits = {}
        self._read_budget_options(budget_option)
        self.window = window
        self.categories = categories        
        self.initialized = False

#    def _isNumerical(self, data, column):
#        """
#        check whether a column in a DataFrame is numerical
#        """
#        return is_numeric_dtype(data[column])#column.dtype)
#
#    def _isCategorical(self, data, column):
#        """
#        check whether a column in a DataFrame is categorical
#        """
#        return not self._isNumerical(data, column)#data.dtypes[column].name == 'category'        
#        

    def _read_budget_options(self, budget_option):
        """
        reads budget_option and translates it into the budget gain options
        """
        self._budget_once = False
        self._budget_batch = False
        self._budget_inst = False
        self._budget_acq = False
        for btime, bgain in budget_option:
            #if not isinstance(bgain, AbstractBudgetGain):
            #    raise ValueError("The budget gain option must be an instance of the AbstractBudgetGain class")
            if btime == 'once':
                self._budget_once = True
                self._bgain_once = bgain
            elif btime == 'batch':
                self._budget_batch = True
                self._bgain_batch = bgain
            elif btime == 'inst':
                self._budget_inst = True
                self._bgain_inst = bgain
            elif btime == 'acq':
                self._budget_acq = True
                self._bgain_acq = bgain

    def _initialize(self, data):
        """
        determines labels and numerical / categorical columns on first run
        execute once before starting process to initialize all dicts and labels
        Sets self.initialized to true
        is separated from __init__ as columns and other only become available once data is provided
        :param data: the data all columns and categories will be based on
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Provided data is not a pandas.DataFrame")

        #set columns
        self.cat_cols = []
        self.num_cols = []
        if self.categories == None:
            self.categories = {}
            for col in data:
                if self._isCategorical(data, col):
                    self.cat_cols.append(col)
                    self.categories[col] = data[col].unique()
                elif self._isNumerical(data, col):
                    self.num_cols.append(col)
        else:
            for col in self.categories:
                #all empty lists are num_cols
                if len(self.categories[col]) == 0:
                    self.num_cols.append(col)
                else:   
                    self.cat_cols.append(col)

        #remove target column
        try:
            self.cat_cols.remove(self.target_col_name)
        except ValueError:
            pass
        try:
            self.num_cols.remove(self.target_col_name)
        except ValueError:
            pass

        #get labels
        if self.categories == None or not self.target_col_name in self.categories.keys():
            self.labels = data[self.target_col_name].unique()
        else:
            self.labels = self.categories[self.target_col_name]

        self._initialize_queried(self.cat_cols + self.num_cols)
        self.initialized = True

    def _get_quality(self, inst, merits, acq_merit=None):
        """
        quality is the average of merits of all features known within an instance
        used by IPF when evaluating an acquisition candidate
        an empty instance returns a quality of 0
        the higher the better
        :param merits: the global feature merits
        :param inst: the instance
        :param acq_merit: the merit of a further acquisition
        leave None if instance quality without additional acquisition is requested
        """
        if acq_merit is None:
            quality = 0
            known_features = 0
        else:
            quality = acq_merit
            known_features = 1
        for key, item in merits.items():
            if not self.isnan(inst[key]):
                known_features += 1
                quality += item
        #prevent 0 / 0
        if known_features == 0:
            return 0
        return quality / known_features

    def _get_miss_merits(self, inst, merits):
        """
        returns all merits of features missing in an instance
        :param merits: the global feature merits
        """
        miss_merits = {}
        for key, item in merits.items():
            if self.isnan(inst[key]):
                miss_merits[key] = item
        return miss_merits
        
    def _get_merits(self, feature_ranks):
        """
        merits are feature_ranks divided by their acquisition cost
        returns a dict with all of them
        """
        merits = {}
        for key, item in feature_ranks.items():
            cost = self.acquisition_costs[key] if key in self.acquisition_costs else 1
            merits[key] = item / cost
        #print(feature_ranks)
        return merits

    @abstractmethod
    def _get_rank_values(self):
        """
        implement method for calculating ranking values for all features as dict here
        """
        pass
    
    @abstractmethod
    def _on_new_batch(self, data):
        """
        this method gets called at the beginning of each new batch
        this is useful for methods that implement a temporary window for instance-wise decision making
        thus allowing to resync the window
        :param data: the data of the framework window
        """
        pass
    
    @abstractmethod
    def _update_window(self, inst):
        """
        this method is called if the acquisition was successful and the label of 
        the corresponding instance is known
        :param inst: the instance in question
        """
        pass
    
    def get_data(self, data):
        """
        gets the data with additionally requested features
        """
        #initialization only possible after representative data in window
        if not self.initialized:                
            self._initialize(self.window.get_window_data())
            if self._budget_once:
                self.budget_manager.add_budget(self._bgain_once())
        
        if hasattr(data, "iterrows"):
            if self._budget_batch:
                self.budget_manager.add_budget(self._bgain_batch())
            
            #since window is batch based, reset temporary window to equal the batch window again
            self._on_new_batch(self.window.get_window_data())
            
            #calculate global feature merits, saved into class to allow statistics
            self.merits = self._get_merits(self._get_rank_values())
            
            for index, row in data.iterrows():
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst())
                #choose candidates
                #candidates[{best, 2nd best,...}][{column, merit}]
                candidates = self._get_miss_merits(inst=row, merits=self.merits)
                candidates = sorted(candidates.items(), key=operator.itemgetter(1), reverse=True)
                
                while len(candidates) > 0:
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    #self.budget_manager.add_budget()
                    quality = self._get_quality(inst=row, merits=self.merits, acq_merit=candidates[0][1])
                    if self.budget_manager.acquire(quality, self.acquisition_costs.get(candidates[0][0], 1)):
                        self.queried[candidates[0][0]] += 1

                        #get feature and replace current inst in iteration and data
                        feature_val = self.get_feature(inst=row, column=candidates[0][0])
                        row[candidates[0][0]] = feature_val
                        data.loc[[index], [candidates[0][0]]] = feature_val

                        if not self.put_back:
                            #break if only most valuable feature is to be acquired
                            break

                        candidates.pop(0)
                    else:
                        #get next instance if most interesting feature not interesting enough
                        break
                
                if self.label_known(inst=row):
                    #update window
                    self._update_window(inst=row)
                    self.merits = self._get_merits(self._get_rank_values())
                    
        else:
            raise ValueError("A pandas.DataFrame expected")

        return data
    
    def get_name(self):
        return "abstract_supervised_merit_ranking"

    def get_stats(self, index=0):
        pd_index = pd.MultiIndex.from_product([[const.active_feature_acquisition_merits, const.active_feature_acquisition_queries], [x for x in self.queried.keys()]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        
        for col, count in self.queried.items():
            stats.loc[index, (const.active_feature_acquisition_queries, col)] = count        
        for col, merit in self.merits.items():
            stats.loc[index, (const.active_feature_acquisition_merits, col)] = merit
        
        return stats
