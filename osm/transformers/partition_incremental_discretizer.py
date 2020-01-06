from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np

class PartitionIncrementalDiscretizer(TransformerMixin):
    def __init__(self, intervals, mins, maxs, alphas, n_bins, strategy="frequency", nan_behavior="ignore"):
        """
        A list of PID discretizers that memorize all seen values, groups them and build the
        discretization from this set
        Discretizes in equal width bins over all seen values
        :param intervals: a list of the amount of invervals for layer 1
        :param mins: a list of the minimum initialized values for layer 1
        :param maxs: a list of the maximum initialized values for layer 1
        :param alphas: a list of split trigger sensitivities that split an interval into two
        :param n_bins: a list of bins created when converting the observations of layer 1 into the discretization process
        :param strategy: how the final bins are setup; either "frequency" or "width"
        :param nan_behavior: how should nan be digitized; only "ignore" defined
        """
        p = len(intervals)
        if len(alphas) != p or len(n_bins) != p:
            raise ValueError("All parameters must be of equal size")
        
        if strategy != "frequency" and strategy != "width":
            raise ValueError("Strategy must be either frequency or width")
        
        self.frequency_bins = strategy == "frequency"
        self.intervals = intervals
        self.mins = mins
        self.maxs = maxs
        #ensure max - min > 0
        for i in range(len(self.mins)):
            if self.mins[i] == self.maxs[i]:
                self.maxs[i] += 1.e-8
        self.alphas = alphas
        self.n_bins = n_bins       
    
    def digitize(self, X):
        if isinstance(X, pd.DataFrame):
            return X.apply(lambda col: col.apply(lambda val: self.pids[col.name].digitize(val))) 
        else:
            for col, val in X.iteritems():
                X[col] = self.pids[col].digitize(val)
            return X
    
    def update_layer1(self, X):
        for col in X:
            for x in X[col]:
                self.pids[col].update_layer1(x)
    
    def fit(self, X, y=None):
        self.pids = {}
        i = 0
        for col in X:
            if col in self.intervals:
                self.pids[col] = _ValPID(self.intervals[col], self.mins[col], self.maxs[col], self.alphas[col], self.n_bins[col])
            else:
                self.pids[col] = _ValPID(self.intervals[i], self.mins[i], self.maxs[i], self.alphas[i], self.n_bins[i])
                i += 1
        
        return self
        
    def transform(self, X):
        #iterate over each column
        #iterate over each value
        
        for col in X:
            for x in X[col]:
                self.pids[col].update_layer1(x)
            if self.frequency_bins:
                self.pids[col].set_equal_frequency_limits()
            else:
                self.pids[col].set_equal_width_limits()
                
            #digitizing
            X[col] = X[col].apply(lambda x: self.pids[col].digitize(x))
        
        return X
        # Xt = check_array(X, copy=True, dtype=FLOAT_DTYPES)        
        # for y in range(Xt.shape[1]):
            # #column = Xt[:, y]
            # #updates
            # for x in column:
                # self.pids[y].update_layer1(x)
            # if self.frequency_bins:
                # self.pids[y].set_equal_frequency_limits()
            # else:
                # self.pids[y].set_equal_width_limits()
            
            # #digitizing
            # for x in range(Xt.shape[0]):
                # Xt[x,y] = self.pids[y].digitize(Xt[x,y])

        # return Xt
        
        
class _ValPID:
    def __init__(self, intervals, min, max, alpha, n_bins):
        """
        An online discretization method based on two layers of frequency counting 
        :param n_bins: the final amount of bins to be produced
        """
        self.min = min
        self.max = max
        self.step = (self.max - self.min) / intervals
        self.n_bins = n_bins
        
        self.nrb = 0 # number of breaks               
        self.alpha = alpha # split threshold 
        self.breaks = [] #[min, max] # interval border values
        
        val = self.min
        while val <= self.max:
            self.breaks.append(val)
            self.nrb += 1
            val += self.step
        self.counts = [0]*self.nrb # number of values within intervals
        self.nr = 0 # number of seen values
        
    #def step(self):
    #    return (self.max - self.min) / (self.nrb + 1)
        
    def update_layer1(self, x):
        """
        updates frequency statistic and bin intervals given the incoming value x
        :param x: the value to be added to the layer
        """
        #skip missing data
        if np.isnan(x):
            return
        #get insertion point
        if x <= self.breaks[0]:
            k = 0
            self.min = x
        elif x >= self.breaks[self.nrb-1]:
            k = self.nrb - 1
            self.max = x
        else:
            k = max(min(2 + int((x - self.breaks[0]) / self.step), self.nrb - 1), 0)
            #if k <= 0 or x < self.breaks[0] or k >= len(self.breaks):
            #    print(str(x) + " " + str(k) + " " + str(self.step))
            #iterate to insertion point
            while x < self.breaks[k]: k -= 1
            while x > self.breaks[k]: k += 1
        
        #print(str(x) + " " + str(k))
        
        self.counts[k] += 1
        self.nr += 1
        
        #splitting trigger
        if (1 + self.counts[k]) / (self.nr + 2) > self.alpha:
            val = self.counts[k] / 2
            self.counts[k] = val
            if k == 0:
                self.breaks.insert(0, self.breaks[0] - self.step)
                self.counts.insert(0, val)
            elif k == self.nrb - 1:
                self.breaks.append(self.breaks[self.nrb - 1] + self.step)
                self.counts.append(val)
            else:
                self.breaks.insert(k, (self.breaks[k - 1] + self.breaks[k]) / 2)
                self.counts.insert(k, val)
            self.nrb += 1

    def set_equal_width_limits(self):
        """
        Creates equal size bins defined by their upper break limits
        """
        self.bin_breaks = []
        
        k = 0
        width = (self.max - self.min) / self.n_bins
        val = self.min + width
        while val <= self.max - width + 1.e-8:
            while self.breaks[k] < val: k += 1
            self.bin_breaks.append(self.breaks[k])            
            val += width
        self.bin_breaks.append(self.max)

    def set_equal_frequency_limits(self):
        """
        Creates equal frequency bins defined by their upper break limits
        """
        self.bin_breaks = []
        
        k = 0
        width = self.nr / self.n_bins
        overhead = 0
        for i in range(self.n_bins - 1):
            val = overhead
            while val < width:
                val += self.counts[k]
                k += 1
            overhead = val - width
            self.bin_breaks.append(self.breaks[k - 1])
        self.bin_breaks.append(self.max)
        
    def digitize(self, x):
        """
        transforms a single value into its discretized state
        must have set bin_breaks beforehand
        """
        if np.isnan(x): return x
        for i in range(self.n_bins): 
            if x < self.bin_breaks[i]:
                return i
        return self.n_bins - 1

    # def fit(self, X, y=None):
        
        # return self

    # def transform(self, X):
        
        # return X
