from abc import ABC, abstractmethod
import pandas as pd
from pandas.api.types import is_numeric_dtype
import math
import operator
import copy
from blist import sorteddict
import osm.data_streams.constants as const

from osm.data_streams.windows.abstract_window import AbstractWindow

from osm.data_streams.active_feature_acquisition.abstract_feature_acquisition import AbstractActiveFeatureAcquisitionStrategy

class AbstractAED(AbstractActiveFeatureAcquisitionStrategy):
    #TODO: integrate differing costs
    #TODO: get function to tell if label known
    #TODO: get the acquisition module
    #TODO: normalization with only 1 value: behavior
    def __init__(self, target_col_name, budget_manager, put_back=False, categories=None, acquisition_costs={}, debug=True):
        """
        Active Feature Acquisition strategy that ranks features accoring to average euclidean distance
        Requires the exact categories and possible category values beforehand
        Abstract basis for other different counting mechanisms AED algorithms might implement
        Implement get_data(self, data) to be used
        :param put_back: whether an instance once acquired a feature will enter the decision process of 
        acquisition again
        :param categories: provide categories for all categorical features as dict 
        empty categorical names are interpreted as numerical columns
        alternatively if left None make initally labeled data in window represent 
        all categorical features at least once 
        """
        if not categories == None and not isinstance(categories, dict):
            raise ValueError("categories is not a dictionary nor left none")
        super().__init__(target_col_name=target_col_name,
                            budget_manager=budget_manager,
                            acquisition_costs=acquisition_costs,
                            put_back=put_back,
                            debug=debug)
        self.categories = categories
        self.initialized = False

    def _isNumerical(self, data, column):
        """
        check whether a column in a DataFrame is numerical
        """
        return is_numeric_dtype(data[column])#column.dtype)

    def _isCategorical(self, data, column):
        """
        check whether a column in a DataFrame is categorical
        """
        return not self._isNumerical(data, column)#data.dtypes[column].name == 'category'

    def _initialize(self, data):
        """
        determines labels and numerical / categorical columns on first run and creates dicts for calculations
        execute once before starting process to initialize all dicts and labels
        Sets self.initialized to true
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

        #init normalize dicts
        self.num_max = {}
        self.num_min = {}

        #init feature dicts
        self.num_counts = {}
        self.num_sums = {}
        self.cat_counts = {}

        #set cat_col dicts
        for col in self.cat_cols:
            self.cat_counts[col] = {}
            for label in self.labels:
                self.cat_counts[col][label] = {}

        #set num_col dicts
        for col in self.num_cols:
            self.num_counts[col] = {}
            self.num_sums[col] = {}

        self._initialize_queried(self.cat_cols + self.num_cols)
        self.initialized = True

    def _normalize(self, min_val, max_val, val, count=1):
        """
        unity-based normalization for a single value
        :param count: for calculating normalized sums 
        """
        return (val - (min_val * count)) / (max_val - min_val)

    """
    def _normalize(self, data, columns=None):
        
        #unity-based normalization bringing numerical features into range [0,1]
        #:param data: the data to be normalized
        #:param columns: the columns to be normalized; set None to use self.num_cols
        
        if columns is None:
            columns = self.num_cols

        norm_data = pd.DataFrame.copy(data, deep=True)
        for col in columns:
            #save num_max and num_min to check for later additions
            if self.keep_normalization:
                self.num_max[col] = max(norm_data[col].max(), self.num_max[col])
                self.num_min[col] = min(norm_data[col].min(), self.num_min[col])
            else:
                self.num_max[col] = norm_data[col].max()
                self.num_min[col] = norm_data[col].min()
            
            if self.num_max[col] != self.num_min[col]:
                norm_data[col] = (norm_data[col] - self.num_min[col]) / (self.num_max[col] - self.num_min[col])
                
        return norm_data
    """

    def _reset_counts(self, data):
        """
        resets sums and counts of data to 0 and counts and adds data
        sets max and min
        """
        #Mean values for each feature given label
        #  sum of features given label for numerical features
        #  count of features given label for numerical features
        #  count of features given label and value for categorical features
        
        #normalize for sum and 0 to 1 distances
        #wd = self._normalize(data=data)

        #set sums and counts back to 0
        for label in self.labels:
            for col in self.num_cols:
                self.num_counts[col][label] = 0
                self.num_sums[col][label] = 0
            for col in self.cat_cols:
                for val in self.categories[col]:
                    self.cat_counts[col][label][val] = 0

        #set max and min
        for col in self.num_cols:
            self.num_max[col] = data[col].max()
            self.num_min[col] = data[col].min()

        #add each row to the counts and sums
        for index, row in data.iterrows():
            #assume each entry in window to have label
            label = row[self.target_col_name]
            for col in self.num_cols:
                #skip nans
                if self.isnan(row[col]):
                    continue
                self.num_counts[col][label] += 1
                self.num_sums[col][label] += row[col]
            for col in self.cat_cols:
                #skip nans
                if self.isnan(row[col]):
                    continue
                val = row[col]
                self.cat_counts[col][label][val] += 1

    def _add_to_counts(self, inst):
        """
        adds an inst to the counting statistics
        :param inst: labeled inst
        """
        label = inst[self.target_col_name]

        #numerical columns
        for col in self.num_cols:
            val = inst[col]
            #skip missing values
            if self.isnan(val):
                continue

            if self.num_min[col] > val: self.num_min[col] = val
            if self.num_max[col] < val: self.num_max[col] = val

            #add to it
            self.num_counts[col][label] += 1
            self.num_sums[col][label] += val

        #categorical columns
        for col in self.cat_cols:
            val = inst[col]
            if not self.isnan(inst[col]):
                self.cat_counts[col][label][val] += 1

    def _get_aed(self):
        """
        returns a dict containing all features and their average euclidean distances
        """

        distances = {}

        #numerical features
        for col in self.num_cols:
            means = {}
            for label in self.labels:
                norm_sum = self._normalize(val=self.num_sums[col][label],
                                            min_val=self.num_min[col],
                                            max_val=self.num_max[col],
                                            count=self.num_counts[col][label])
                means[label] = norm_sum / self.num_counts[col][label]
            
                #handle complete nan classes by removing them
                if self.isnan(means[label]):
                    means.pop(label)

            #TODO: possibly more elegant solution? itertools?
            means2 = copy.deepcopy(means)
            squared_sums = 0
            for dkey, dvalue in means.items():
                means2.pop(dkey)
                for dkey2, dvalue2 in means2.items():
                    squared_sums += (dvalue - dvalue2) ** 2
            
            distances[col] = math.sqrt(squared_sums)

        #categorical features
        for col in self.cat_cols:
            #get count of features per label
            L = {}
            for label in self.labels:
                L[label] = sum(self.cat_counts[col][label].values())

                #handle no occurences of features for class by popping label from dict
                if L[label] == 0:
                    L.pop(label)
            
            L2 = copy.deepcopy(L)
            label_sums = 0
            for l1 in L.keys():
                L2.pop(l1)
                for l2 in L2.keys():
                    value_sums = 0
                    for v in self.categories[col]:
                        value_sums += math.sqrt(((self.cat_counts[col][l1][v] / L[l1]) - (self.cat_counts[col][l2][v] / L[l2])) ** 2)

                    label_sums += value_sums / len(self.categories[col])
            
            distances[col] = label_sums

        
        return distances

    def _get_quality(self, inst, aeds, merit):
        """
        return the quality of a single data row
        the higher the better
        """
        quality = 0
        known_features = 1
        for key, item in aeds.items():
            if not self.isnan(inst[key]):
                cost = self.acquisition_costs[key] if key in self.acquisition_costs else 1
                known_features += 1
                quality += item / cost
        #prevent 0 / 0
        #if known_features == 0:
        #    return 0
        return quality / known_features

    def _get_merits(self, inst, aeds):
        """
        return the merits of all unknown features
        """
        #TODO: look into better cost integration
        merits = {}
        for key, item in aeds.items():
            if self.isnan(inst[key]):
                cost = self.acquisition_costs[key] if key in self.acquisition_costs else 1
                merits[key] = item / cost
        return merits

    def get_name(self):
        return "abstract_average_euclidean_distance"

    def get_stats(self, index=0):
        pd_index = pd.MultiIndex.from_product([[const.active_feature_acquisition_stats], [x for x in self.queried.keys()]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        
        for col, count in self.queried.items():
            stats.loc[index, (const.active_feature_acquisition_stats, col)] = count        
        
        return stats
        
class MultiWindowAED(AbstractAED):
    #TODO: make ild data instead of window
    #TODO: make windows not list based but window based
    # 
    # self.labels
    # self.categories[col]
    # 
    # self.cat_cols
    # self.num_cols
    #
    # self.num_min[col]
    # self.num_max[col]
    # self.num_counts[col][label]
    # self.num_sums[col][label]
    # 
    # self.cat_counts[col][label][value]
    #
    # self.windows[col][label]           
    # self.windows_counts[col][label]
    # 
    def __init__(self, ild, window_size, target_col_name, budget_manager, put_back=False, categories=None, acquisition_costs={}, debug=True):
        """
        Active Feature Acquisition strategy that ranks features accoring to average euclidean distance
        Requires the exact categories and possible category values beforehand
        
        Implements simple individual sliding windows for each label column combination
        that limit the amount of recent values available for AED calculation
        :param ild: set as same window as used in framework for initially labeled data
        Assumes it to be feature complete
        :param put_back: whether an instance once acquired a feature will enter the decision process of 
        acquisition again
        :param categories: provide categories for all categorical features as dict 
        empty categorical names are interpreted as numerical columns
        alternatively if left None make initally labeled data in window represent 
        all categorical features at least once 
        """
        super().__init__(target_col_name=target_col_name,
                            put_back=put_back,
                            budget_manager=budget_manager,
                            acquisition_costs=acquisition_costs,
                            categories=categories,
                            debug=debug)
        self.ild = ild
        self.window_size = window_size
        #self._initialize(ild)
    
    def _initialize(self, data):
        """
        determines labels and numerical / categorical columns on first run and creates dicts for calculations
        execute once before starting process to initialize all dicts and labels
        Sets self.initialized to true
        Sets up windows
        :param data: the data all columns and categories will be based on
        """
        super()._initialize(data)

        #set windows
        self.windows = {}
        self.windows_counts = {}
        for col in self.cat_cols + self.num_cols:
            self.windows[col] = {}
            self.windows_counts[col] = {}
            #for label in self.labels:
            #    self.windows[col][label] = []
            #    self.windows_counts[col][label] = 0

    def _reset_counts(self, data):
        """
        resets sums and counts of data to 0 and recounts and resums all
        occurences for the given data
        adds all data into windows
        data is normalized after receiving it
        """
        #first entry - set up num_min and num_max
        first = data.iloc[0]
        for col in self.num_cols:
            self.num_min[col] = first[col]
            self.num_max[col] = first[col]
        
        #reset all counts, sum and windows to 0
        for label in self.labels:
            for col in self.cat_cols:
                self.windows[col][label] = []
                self.windows_counts[col][label] = 0
                for val in self.categories[col]:
                    self.cat_counts[col][label][val] = 0
            for col in self.num_cols:
                self.num_sums[col][label] = 0
                self.num_counts[col][label] = 0
                self.windows[col][label] = []
                self.windows_counts[col][label] = 0

        #add all initial data to windows
        for index, row in data.iterrows():
            self._add_to_counts(row)
        
    def _add_to_counts(self, inst):
        """
        adds an instance to the counting statistics
        :param inst: labeled inst
        """
        label = inst[self.target_col_name]

        #numerical columns
        for col in self.num_cols:
            val = inst[col]
            #skip missing values
            if self.isnan(val):
                continue

            if self.num_min[col] > val: self.num_min[col] = val
            if self.num_max[col] < val: self.num_max[col] = val

            #add to it
            self.windows[col][label].append(val)
            self.windows_counts[col][label] += 1
            self.num_counts[col][label] += 1
            self.num_sums[col][label] += val

            #check for windows
            if self.windows_counts[col][label] > self.window_size:
                val = self.windows[col][label][0]

                self.windows[col][label].pop(0)
                self.windows_counts[col][label] -= 1
                self.num_counts[col][label] -= 1
                self.num_sums[col][label] -= val

                #handle new normalization
                if self.num_min[col] == val:
                    min_val = self.num_min[col]
                    for l in self.windows[col].keys():
                        if min(self.windows[col][l]) < min_val: min_val = min(self.windows[col][l])
                    self.num_min[col] = min_val

                if self.num_max[col] == val: 
                    max_val = self.num_max[col]
                    for l in self.windows[col].keys():
                        if max(self.windows[col][l]) > max_val: max_val = max(self.windows[col][l])
                    self.num_max[col] = max_val

        #categorical columns
        for col in self.cat_cols:
            val = inst[col]
            if not self.isnan(inst[col]):
                self.cat_counts[col][label][val] += 1
                self.windows[col][label].append(val)
                self.windows_counts[col][label] += 1

                #check for windows
                if self.windows_counts[col][label] > self.window_size:
                    self.windows_counts[col][label] -= 1
                    self.cat_counts[col][label][val] -= 1
                    self.windows[col][label].pop(0)      

    def get_data(self, data):
        #set up for first use
        if not self.initialized:
            self._initialize(self.ild.get_window_data())
            self._reset_counts(self.ild.get_window_data())

        if hasattr(data, "iterrows"):
            #calculate individual feature distances
            distances = self._get_aed()

            for index, row in data.iterrows():
                #get individual merits of missing columns
                #also used to keep track of missing values left
                merits = self._get_merits(inst=row, aeds=distances)
                merits = sorted(merits.items(), key=operator.itemgetter(1), reverse=True)

                #may feed instance back after acquisition
                while len(merits) > 0:
                    self.budget_manager.add_budget()
                    quality = self._get_quality(inst=row, aeds=distances, merit=merits[0][1])
                    if self.budget_manager.acquire(quality,#quality + merits[0][1],
                                                   self.acquisition_costs.get(merits[0][0], 1)):
                        self.queried[merits[0][0]] += 1

                        #get feature and replace current inst in iteration and data
                        feature_val = self.get_feature(inst=row, column=merits[0][0])
                        row[merits[0][0]] = feature_val
                        data.loc[[index], [merits[0][0]]] = feature_val

                        if not self.put_back:
                            #break if only most valuable feature to be acquired
                            break

                        merits.pop(0)
                    else:
                        #get next instance if most interesting feature not interesting enough
                        break
                
                if self.label_known(inst=row):
                    self._add_to_counts(inst=row)
                    distances = self._get_aed()

        else:
            raise ValueError("A pandas.DataFrame expected")

        return data

    def get_name(self):
        return "multi_window_average_euclidean_distance"

class SingleWindowAED(AbstractAED):
    def __init__(self, window, target_col_name, budget_manager, put_back=False, categories=None, acquisition_costs={}, debug=True):
        """
        Active Feature Acquisition strategy that ranks features accoring to average euclidean distance
        Requires the exact categories and possible category values beforehand
        :param window: set as same window as used in framework
        :param put_back: whether an instance once acquired a feature will enter the decision process of 
        acquisition again
        :param categories: provide categories for all categorical features as dict 
        alternatively if left None make initally labeled data in window represent 
        all categorical features at least once 
        """
        super().__init__(target_col_name=target_col_name,
                            put_back=put_back, 
                            budget_manager=budget_manager, 
                            acquisition_costs=acquisition_costs,
                            categories=categories,
                            debug=debug)
        if not isinstance(window, AbstractWindow):
            raise ValueError("The window must be the same instance of AbstractWindow as the framework uses")

        self.ild = window

    def get_data(self, data):
        #initialization only possible after representative data in window
        if not self.initialized:
            self._initialize(self.ild.get_window_data())      

        if hasattr(data, "iterrows"):
            #update counts and sums with labeled data in window
            self._reset_counts(self.ild.get_window_data())

            #calculate individual feature distances
            distances = self._get_aed()

            for index, row in data.iterrows():
                #get individual merits of missing columns
                #also used to keep track of missing values left
                merits = self._get_merits(inst=row, aeds=distances)
                merits = sorted(merits.items(), key=operator.itemgetter(1), reverse=True)

                #update_counts = False

                #may feed instance back after acquisition
                while len(merits) > 0:
                    self.budget_manager.add_budget()
                    quality = self._get_quality(inst=row, aeds=distances, merit=merits[0][1])
                    if self.budget_manager.acquire(quality,#quality + merits[0][1],
                                                   self.acquisition_costs.get(merits[0][0], 1)):
                        #update_counts = True
                        self.queried[merits[0][0]] += 1

                        #get feature and replace current inst in iteration and data
                        feature_val = self.get_feature(inst=row, column=merits[0][0])
                        row[merits[0][0]] = feature_val
                        data.loc[[index], [merits[0][0]]] = feature_val

                        if not self.put_back:
                            #break if only most valuable feature to be acquired
                            break

                        merits.pop(0)
                    else:
                        #get next instance if most interesting feature not interesting enough
                        break
                
                if self.label_known(inst=row):# and update_counts:
                    self._add_to_counts(inst=row)
                    distances = self._get_aed()

        else:
            raise ValueError("A pandas.DataFrame expected")

        return data

    def get_name(self):
        return "single_window_average_euclidean_distance"