from abc import ABC, abstractmethod
import pandas as pd
from pandas.api.types import is_numeric_dtype
import math
import operator
import copy
import osm.data_streams.constants as const

from osm.data_streams.windows.abstract_window import AbstractWindow

from osm.transformers.partition_incremental_discretizer import PartitionIncrementalDiscretizer
from sklearn.pipeline import Pipeline
from additions.transformers.CatImputer import CategoricalRememberingNaNImputer

from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.supervised_merit_ranking import AbstractSMR

#IG(Y|X)    = H(Y) - H(Y|X)
# H(Y)      = Sum(y € Y)(-P(Y=y) * log(P(Y=y)))
#  P(Y=y)   = |Y=y| / |Y|
# H(Y|X)    = Sum(v € X)(P(v) * H(Y|X=v))
#  P(v)     = |X=v| / |X|
#  H(Y|X=v) = Sum(y € Y)(-P(Y=y|X=v) * log(P(Y=y|X=v)))
#
# |Y| -> len(Window)
# |X| -> X count
# y count
# x count 
# y count|X=v
#
# self.cat_count[col][label][value] = ...
# self.lab_count[label] = ...
# self.cat_probs[col][value] = ...
# self.lab_probs[label] = ...
# self.cat_entropies[col] = ...
# self.lab_entropy = ...

class AbstractEntropy(AbstractSMR):
    def __init__(self, window, target_col_name, budget_manager, put_back=False, categories=None, acquisition_costs={}, debug=True):
        """
        Abstract Active Feature Acquisition strategy that ranks features accoring to some entropy metric
        Requires the exact categories and possible category values beforehand
        Utility class for further entropy based ranking methods used for AFA
        :param window: the framework window
        :param put_back: whether an instance once acquired a feature will enter the decision process of 
        acquisition again
        :param categories: provide categories for all categorical features as dict 
        empty categorical names are interpreted as numerical columns
        alternatively if left None make initally labeled data in window represent 
        all categorical features at least once 
        """
        super().__init__(target_col_name=target_col_name,
                            window=window,
                            budget_manager=budget_manager,
                            acquisition_costs=acquisition_costs,
                            put_back=put_back,
                            categories=categories,
                            debug=debug)
        self.initialized = False

    def _initialize(self, data):
        """
        determines labels and numerical / categorical columns on first run and creates dicts for calculations
        execute once before starting process to initialize all dicts and labels
        Sets self.initialized to true
        :param data: the data all columns and categories will be based on
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Provided data is not a pandas.DataFrame")

        super()._initialize(data)
        
        #if not self.num_cols:
        #    raise ValueError("Entropy methods are not usable on continous data")
        
        #prepare discretizer for num values
        self.pid_cols = self.num_cols
        intervals = 200
        bins = 8
        alphas = 0.25
        strategy = "frequency"
        n = len(self.num_cols)
        mins = list(data[self.pid_cols].min())
        maxs = list(data[self.pid_cols].max())
        
        self.pid = PartitionIncrementalDiscretizer(
                                intervals=[intervals]*n, 
                                mins = mins,
                                maxs= maxs,
                                alphas=[alphas]*n, 
                                n_bins=[bins]*n,
                                strategy=strategy)        
        
        self.pid.fit(data[self.pid_cols])
        self.pid.transform(data[self.pid_cols])
        
        #alter num and cat cols to support default smr methods
        self.num_cols = []
        self.cat_cols += self.pid_cols
        for col in self.pid_cols:
            self.categories[col] = list(range(bins))
        
        self.imp = CategoricalRememberingNaNImputer(categories=self.categories)
        self.pipeline = Pipeline(steps=[
            ('discretizer', self.pid),
            ('imputer', self.imp)])
        
        #init feature dicts
        self.cat_counts = {}
        self.lab_counts = {}
        self.cat_probs = {}
        self.lab_probs = {}
        self.cat_entropies = {}
        self.lab_entropy = 0
        
        #set cat_col dicts
        for col in self.cat_cols:
            self.cat_counts[col] = {}
            self.cat_probs[col] = {}
            for label in self.labels:
                self.cat_counts[col][label] = {}         
    
    def log(self, x, base=2):
        """
        returns math.log(x, base) and returns 0
        if x == 0        
        """
        if x == 0:
            return 0
        return math.log(x, base)
    
    def _calculate_probs_and_entropy_y(self):
        """
        calculate y probabilities and H(Y) using the
        self.count statistics and saves it in
        self.lab_probs and self.lab_entropy
        """
        #calculate y probabilities and H(Y)
        #H(Y) = Sum(y € Y)(-P(Y=y) * log(P(Y=y)))
        self.lab_entropy = 0
        s = sum(self.lab_counts.values())
        for label, count in self.lab_counts.items():
            self.lab_probs[label] = count / s
            self.lab_entropy -= self.lab_probs[label] * self.log(self.lab_probs[label])
    
    def _calculate_probs_and_entropy_x(self, columns):
        """
        calculates x probabilities and H(Xi) using the
        self.count statistics and saves it in
        self.cat_probs and self.cat_entropies
        """
        #calculate x probabilities and H(Xi)
        #H(Xi) = Sum(x € Xi)(-P(Xi=x) * log(P(Xi=x)))
        for col in columns:
            self.cat_entropies[col] = 0
            xsum = 0
            for val in self.categories[col]:
                self.cat_probs[col][val] = 0
                for label in self.labels:
                    self.cat_probs[col][val] += self.cat_counts[col][label][val]
                xsum += self.cat_probs[col][val]
            for val in self.categories[col]:
                self.cat_probs[col][val] /= xsum
                self.cat_entropies[col] -= self.cat_probs[col][val] * self.log(self.cat_probs[col][val])
    
    def _calculate_probs_and_entropies(self):
        """
        calculates all entropies and probabilities using the
        self.count statistics and saves it in
        self.lab_probs, self.lab_entropy, self.cat_probs and self.cat_entropies
        """
        self._calculate_probs_and_entropy_y()
        self._calculate_probs_and_entropy_x(self.cat_cols)
    
    def _get_info_gain(self, column):
        """
        """
        #IG(Y|X)    = H(Y) - H(Y|X)
        #H(Y|X)     = Sum(x € X)(p(x) * H(Y|X=x))
        #p(x)       = |X=x| / |X|
        #H(Y|X=x)   = -Sum(y € Y)(P(Y=y|X=x) * log(P(Y=y|X=x)))
        #P(Y=y|X=x) = |Y=y|X=x| / |X=x|
        
        #+IG(Y|X)
        #+  H(Y)
        #+    P(Y=y)
        #+      |Y=y|
        #+      |Y|
        #+  H(Y|X)
        #+    p(x)
        #+      |X=x|
        #+      |X|
        #+    H(Y|X=x)
        #+      P(Y=y|X=x)
        #+        |Y=y|X=x|
        #+        |X=x|
                
        #xsum = sum(x for labs in self.cat_probs[column].values() for x in labs.values())
        #|X|
        xcounts = {}
        for val in self.categories[column]:
            xcounts[val] = 0
        
        #|X=x|
        xsum = 0
        for label in self.cat_counts[column]:
            for val, xcount in self.cat_counts[column][label].items():
                xcounts[val] += xcount
                xsum += xcount
        
        #H(Y|X)
        cond_ent = 0        
        for val in self.cat_counts[column][label]:
            if xcounts[val] == 0: continue
            #p(x)
            px = xcounts[val] / xsum
            #H(Y|X=x)
            hyx = 0
            for label in self.labels:
                #P(Y=y|X=x)
                pyx = self.cat_counts[column][label][val] / xcounts[val]
                hyx -= pyx * self.log(pyx)
            cond_ent += px * hyx
            
        #IG(Y|X)    
        return self.lab_entropy - cond_ent
        
    def _get_symmetric_uncertainty(self, column):
        """
        """
        #SU(X,Y)    = (IG(X|Y)) / (H(X) + H(Y))
        #IG(X|Y)    = H(X) - H(X|Y)
        #H(X|Y)     = Sum(y € Y)(p(y) * H(X|Y=y))
        #p(y)       = |Y=y| / |Y|
        #H(X|Y=y)   = -Sum(x € X)(P(X=x|Y=y) * log(P(X=x|Y=y)))
        #P(X=x|Y=y) = |X=x|Y=y| / |Y=y|
        
        #+SU(X,Y)
        #+  IG(X|Y)
        #+    H(X)
        #+    H(X|Y)
        #+      p(y)
        #+      H(X|Y=y)
        #+        P(X=x|Y=y)
        #+          |X=x|Y=y|
        #+          |Y=y|
        #+  H(X)
        #+    P(X=x)
        #+      |X=x|
        #+      |X|
        #+  H(Y)
        #+    P(Y=y)
        #+      |Y=y|
        #+      |Y|
        """        
        cond_ent = 0
        for label in self.lab_counts:
            if self.lab_counts[label] == 0: continue
            #p(y)
            py = self.lab_probs[label]
            #H(X|Y=y)
            hxy = 0
            for val in self.categories[column]:
                #P(X=x|Y=y)
                pxy = self.cat_counts[column][label][val] / self.lab_counts[label]
                hxy -= pxy * self.log(pxy)
            cond_ent += py * hxy
        
        #IG(X|Y)
        info_gain = self.cat_entropies[column] - cond_ent
        
        #SU(X,Y)
        return info_gain / (self.cat_entropies[column] + self.lab_entropy)
        """
        return self._get_info_gain(column) / (self.cat_entropies[column] + self.lab_entropy)
        
    def _on_new_batch(self, data):
        """
        resets counts of data to 0 and counts and adds data
        """
        data[self.pid_cols] = self.pid.digitize(data[self.pid_cols])
        #set counts back to 0
        for label in self.labels:
            self.lab_counts[label] = 0        
        for col in self.cat_cols:
            for label in self.labels:
                for val in self.categories[col]:
                    self.cat_counts[col][label][val] = 0
                
        #add each row to the counts
        for index, row in data.iterrows():
            label = row[self.target_col_name]
            self.lab_counts[label] += 1
            
            for col in self.cat_cols:
                #skip nans
                if self.isnan(row[col]):
                    continue
                val = row[col]
                self.cat_counts[col][label][val] += 1
    
        self._calculate_probs_and_entropies()
    
    def _update_window(self, inst):
        """
        adds an instance to the count variable and automatically
        adjusts the probabilities and entropies of those affected
        """
        inst[self.pid_cols] = self.pid.digitize(inst[self.pid_cols])
        label = inst[self.target_col_name]

        self.lab_counts[label] += 1
        self._calculate_probs_and_entropy_y()
                
        #categorical columns
        for col in self.cat_cols:
            if not self.isnan(inst[col]):
                val = inst[col]
                self.cat_counts[col][label][val] += 1
                self._calculate_probs_and_entropy_x([col])
                
    def get_data(self, data):
        """
        Do usual routine but also update discretizer
        """
        data = super().get_data(data)
        self.pid.update_layer1(data[self.pid_cols])
        return data
        
    def get_name(self):
        return "abstract_entropy"
            
class SingleWindowIG(AbstractEntropy):
    def __init__(self, window, target_col_name, budget_manager, put_back=False, categories=None, acquisition_costs={}, debug=True):
        """
        Active Feature Acquisition strategy that ranks features accoring to information gain
        Requires the exact categories and possible category values beforehand
        :param window: the framework window
        :param put_back: whether an instance once acquired a feature will enter the decision process of 
        acquisition again
        :param categories: provide categories for all categorical features as dict 
        empty categorical names are interpreted as numerical columns
        alternatively if left None make initally labeled data in window represent 
        all categorical features at least once 
        """
        super().__init__(target_col_name=target_col_name,
                            window=window,
                            budget_manager=budget_manager,
                            acquisition_costs=acquisition_costs,
                            put_back=put_back,
                            categories=categories,
                            debug=debug)
        self.initialized = False        
    
    def _get_rank_values(self):
        """
        returns a dict containing all features and their information gains
        """
        
        info_gains = {}
        
        #caluclate info gain
        for col in self.cat_cols:
            info_gains[col] = self._get_info_gain(col)
            
        return info_gains
        
    def get_name(self):
        return "single_window_information_gain"

class SingleWindowSU(AbstractEntropy):
    def __init__(self, window, target_col_name, budget_manager, put_back=False, categories=None, acquisition_costs={}, debug=True):
        """
        Active Feature Acquisition strategy that ranks features accoring to symmetric uncertainty
        Requires the exact categories and possible category values beforehand
        :param window: the framework window
        :param put_back: whether an instance once acquired a feature will enter the decision process of 
        acquisition again
        :param categories: provide categories for all categorical features as dict 
        empty categorical names are interpreted as numerical columns
        alternatively if left None make initally labeled data in window represent 
        all categorical features at least once 
        """
        super().__init__(target_col_name=target_col_name,
                            window=window,
                            budget_manager=budget_manager,
                            acquisition_costs=acquisition_costs,
                            put_back=put_back,
                            categories=categories,
                            debug=debug)
        self.initialized = False        
    
    def _get_rank_values(self):
        """
        returns a dict containing all features and their symmetric uncertainties
        """
        
        symmetric_uncertainties = {}
        
        #caluclate info gain
        for col in self.cat_cols:
            symmetric_uncertainties[col] = self._get_symmetric_uncertainty(col)
            
        return symmetric_uncertainties
        
    def get_name(self):
        return "single_window_symmetric_uncertainty"    
        