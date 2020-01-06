import pandas as pd
import numpy as np
import sys
import os

import additions.constants as consts

class DatasetGenerator:

    def __init__(self, label_function:str, features:int, indicatives:int, allow_less_indicatives:bool, function_width_mean:int, function_width_variance:int, concepts:int, ensure_new_concepts:bool ):#, labels:int):
        """
        :param label_function: the boolean function that generates the data
        set to "or", "and" or "equiv"
        :param features: the total sum of features
        :param indicatives: the amount of features building the label
        :param allow_less_indicatives: whether there may be fewer than indicatives features building the label
        :param function_width_mean: the average length of a set label function until new label function
        :param function_width_variance: the variance in length of a set label function until new label function
        :param concepts: how many different label functions will be used        
        :param ensure_new_concepts: if set true will not allow a randomly picked concept to be identical to previous concept
        """
        #:param labels: how many different labels the label function can return
        
        if features < 1:
            raise ValueError("At least one feature is required")

        if features < indicatives:
            raise ValueError("Can't have more indicative features than features are available")
            
        if function_width_mean - abs(function_width_variance) < 1:
            raise ValueError("The label function must at least generate one feature")

        if concepts < 1:
            raise ValueError("At least one concept must happen")

        
        #if labels < 2:
        #    raise ValueError("The label function must at least be capable of generating two labels")

        if label_function != "or" and label_function != "and" and label_function != "equiv" and label_function != "xor":
            raise ValueError("The label function is undefined")
            
        if ensure_new_concepts and concepts > 1 and (features == 1 or (not allow_less_indicatives and features == indicatives)):
            raise ValueError("Cannot ensure new concept on second iteration. Parameters only allow for one specific concept!")
            
        self.label_function = label_function        
        self.features = features                       
        self.indicatives = indicatives
        self.allow_less_indicatives = allow_less_indicatives
        self.function_width_mean = function_width_mean
        self.function_width_variance = function_width_variance
        self.concepts = concepts
        self.ensure_new_concepts = ensure_new_concepts
        
    def or_func(self, df, keys):
        for key in keys:
            if df[key]:
                return 1
        return 0
        
    def and_func(self, df, keys):
        for key in keys:
            if not df[key]:
                return 0
        return 1
    
    def equiv_func(self, df, keys):
        val = df[keys[0]]
        for key in keys:
            if val != df[key]:
                return 0
        return 1
        
    def occur_func(self, df, keys, min_times:int, max_times:int):
        if max_times < min_times:
            return False
        occ = 0        
        for key in keys:
            if df[key]:
                occ += 1
                if occ > max_times:
                    return False
        return occ >= min_times
        
    def get_random_dets(self, features=None, indicatives=None, allow_less_indicatives=None):
        features = features or self.features
        indicatives = indicatives or self.indicatives
        allow_less_indicatives = allow_less_indicatives or self.allow_less_indicatives
        return np.unique(np.random.choice(range(features), indicatives, allow_less_indicatives)) # [d0, d1, ..., di] -> y
        
    def generate_dataset(self):
        """
        Returns a new data set from the given parameters
        """
        
        ret_data = pd.DataFrame()
        ret_insts = []
        ret_dets = []
        
        last_dets = []
        
        for c in range(self.concepts):
            print("Generating concept " + str(c))
            data = pd.DataFrame()
            
            #randomize features and function
            insts = self.function_width_mean + int(np.random.uniform(-self.function_width_variance, self.function_width_variance + 1))
            
            dets = self.get_random_dets()
            if self.ensure_new_concepts:
                while set(dets) == set(last_dets):
                    dets = self.get_random_dets()                
                last_dets = dets            
                
            for feature in range(self.features):
                data[str(feature)] = list(map(int, np.random.uniform(0, 2, insts)))            
            
            #generate label
            if self.label_function == "or":
                data["label"] = data.apply(self.or_func, axis=1, keys=dets)                
                #data["label"] = data.apply(func=self.occur_func, axis=1, keys=dets, min_times=1, max_times=len(dets))
            elif self.label_function == "and":
                data["label"] = data.apply(self.and_func, axis=1, keys=dets)
                #data["label"] = data.apply(func=self.occur_func, axis=1, keys=dets, min_times=len(dets), max_times=len(dets))
            elif self.label_function == "equiv":
                data["label"] = data.apply(func=self.equiv_func, axis=1, keys=dets)
            elif self.label_function == "xor":
                data["label"] = data.apply(func=self.occur_func, axis=1, keys=dets, min_times=1, max_times=1)
            
            ret_data = ret_data.append(data)
            ret_insts.append(insts)
            ret_dets.append(dets)
        
        #convert all 0s and 1s to bools to avoid OneHot numerical issues and nan insertion
        ret_data.replace([0, 1],['f', 't'], inplace=True)
        ret_data.reset_index(inplace=True, drop=True)
        return ret_data, ret_insts, ret_dets
        
if __name__ == "__main__":
    """
    Generates a data set according to the following parameters
    - 'and', 'or' or 'equiv' as label function
    - f features
    - x indicative features optionally add '-' to allow for less indicative features to be accepted
    - i mean instances per concept
    - v variance per concept
    - c concepts optionally add '!' to enforce a new concept each next concept
    - if desired, an alternative, python ready path
    example: "and f6 x3- i400 v100 c10!"
    which yields a dataset with 10 changing concepts each of a length between 300 and 500 instances
    each instance consists of 6 features with up to 3 determining the corresponding label through an
    and function
    """
    label_function = "equiv"
    features = 10
    indicatives = 3
    allow_less_indicatives = True    
    function_width_mean = 25
    function_width_variance = 0
    concepts = 3
    ensure_new_concepts = False
    path = consts.DIR_GEN
    
    for arg in sys.argv[1:]:
        if arg == 'or' or arg == 'and' or arg == 'equiv':
            label_function = arg
        elif '\\' in arg:
            if os.path.exists(arg):
                path = arg
            else:
                raise ValueError("Path {} does not exist".format(arg))
            
        elif 'f' in arg:
            features = int(arg.replace('f', ''))
        elif 'x' in arg:
            val = arg.replace('x', '')
            allow_less_indicatives = '-' in val
            indicatives = int(val.replace('-', ''))
        elif 'i' in arg:
            function_width_mean = int(arg.replace('i', ''))
        elif 'v' in arg:
            function_width_variance = int(arg.replace('v', ''))
        elif 'c' in arg:
            val = arg.replace('c', '')
            ensure_new_concepts = '!' in val
            concepts = int(val.replace('!', ''))        
            
    generator = DatasetGenerator(label_function=label_function, 
                                 features=features, 
                                 indicatives=indicatives, 
                                 allow_less_indicatives=allow_less_indicatives,
                                 function_width_mean=function_width_mean,
                                 function_width_variance=function_width_variance,
                                 concepts=concepts,
                                 ensure_new_concepts=ensure_new_concepts)
    df, insts, dets = generator.generate_dataset()       
    
    dirname = "gen_" + label_function + "_" + str(features) + "_" + str(indicatives) + "_" + str(allow_less_indicatives) + "_" + str(function_width_mean) + "_" + str(function_width_variance) + "_" + str(concepts) + "_" + str(ensure_new_concepts)
    path = os.path.join(path, dirname)
    if not os.path.exists(path):
        os.mkdir(path)    
    df.to_csv(os.path.join(path, "raw_data.csv"), index=False)
    df.to_pickle(os.path.join(path, "raw_data.pkl.gzip"))
    
    f = open(os.path.join(path, "raw_data.meta"), "w")
    f.write("iteration\tinsts\tdets\n")
    for i in range(concepts):
        f.write(str(i) + '\t' + str(insts[i]))
        for det in dets[i]:
            f.write('\t' + str(det))
        f.write('\n')
    f.close()
    
    f = open(os.path.join(path, "params.txt"), "w")
    f.write("label_function\t" + str(label_function) + '\n')
    f.write("features\t" + str(features) + '\n')
    f.write("indicatives\t" + str(indicatives) + '\n')
    f.write("allow_less_indicatives\t" + str(allow_less_indicatives) + '\n')
    f.write("function_width_mean\t" + str(function_width_mean) + '\n')
    f.write("function_width_variance\t" + str(function_width_variance) + '\n')
    f.write("concepts\t" + str(concepts) + '\n')
    f.write("ensure_new_concepts\t" + str(ensure_new_concepts) + '\n')
    f.close()