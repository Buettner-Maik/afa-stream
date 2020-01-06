import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../osm'))

import additions.constants as consts
import pandas as pd
import multiprocessing as mp
from numpy import array
from datetime import datetime

from additions.transformers.MissTransformer import MissTransformer
from additions.transformers.CatImputer import CategoricalRememberingNaNImputer

from osm.data_streams.algorithm.framework import FrameWork

from osm.data_streams.active_learner.strategy.pool_based.no_active_learner import NoActiveLearner

from osm.data_streams.oracle.simple_oracle import SimpleOracle

from osm.data_streams.windows.sliding_window import SlidingWindow
from osm.data_streams.windows.fixed_length_window import FixedLengthWindow

from osm.data_streams.budget_manager.incremental_percentile_filter import IncrementalPercentileFilter
from osm.data_streams.budget_manager.simple_budget_manager import SimpleBudgetManager
from osm.data_streams.budget_manager.no_budget_manager import NoBudgetManager

from osm.data_streams.active_feature_acquisition.no_feature_acquisition import NoActiveFeatureAcquisition
from osm.data_streams.active_feature_acquisition.random_acquisition import RandomFeatureAcquisition
from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.average_euclidean_distance import SingleWindowAED, MultiWindowAED
from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.entropy_based import SingleWindowIG, SingleWindowSU

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from additions.sgd_predict_proba_fix import SGDPredictFix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder#, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class Dataset():
    """
    container of all information needed for a run of a categorical data set
    can run an analysis with the the provided osm framework
    """
    def __init__(self, name:str, directory:str, filename:str, 
                 cat_cols:list, num_cols:list, targetcol:str,
                 pid_mins:list = [], pid_maxs:list = [], categories = None):
        """
        saves all meta data for a dataset
        :param name: the name of the dataset
        :param directory: the directory of the raw data
        :param filename: the name of the raw data file
        :param cat_cols: all columns with categorical data except the target column to be considered
        :param num_cols: all columns with numerical data except the target column to be considered
        :param targetcol: the target column containing the labels
        :param pid_mins: the min values the PID is initialized
        :param pid_maxs: the max values the PID is initialized
        :param categories: provide all categorical values for each categorical column
        alternatively leave None to automatically get values
        """
        self.name = name
        self.directory = directory
        self.filename = filename
        
        #remove targetcol from cols to be considered
        try:
            cat_cols.remove(targetcol)
        except ValueError:
            pass
        try:
            num_cols.remove(targetcol)
        except ValueError:
            pass

        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.targetcol = targetcol
        self.pid_mins = pid_mins
        self.pid_maxs = pid_maxs
        if categories is None:
            self.categories = self._get_categories_list(df=pd.read_pickle(directory + 'raw_data.pkl.gzip'))
        else:
            self.categories = categories

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def summary_str(self, pre_folder="", i_str="", m_str=""):
        """
        get a summary string
        """
        filepath = self.directory
        
        if pre_folder: filepath = filepath + pre_folder + '/'
        if not i_str: filepath = filepath + 'default/'
        else: filepath = filepath + i_str + '/'
        if not m_str: filepath = filepath + 'default/summary.pkl.gzip'
        else: filepath = filepath + m_str + '/summary.pkl.gzip'
        return filepath     

    def get_default_pipeline(self):
        """
        returns the default pipeline
        """
        cat_trans = Pipeline(steps=[
            ('imputer', CategoricalRememberingNaNImputer(
                categories=self._get_categories_dict())),
            ('encoder', OneHotEncoder(#handle_unknown='ignore', 
                                      categories=self.categories,
                                      sparse=False))
        ])
        num_trans = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_trans, self.num_cols),
            ('cat', cat_trans, self.cat_cols)
        ])

        feature_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
        return feature_pipeline
        
    def load_df(self):
        df = pd.read_csv(self.directory + self.filename, index_col=False)
        if any(self.cat_cols):
            cats = df[self.cat_cols]#.astype('object')
        else:
            cats = pd.DataFrame()
        if any(self.num_cols):
            nums = df[self.num_cols].astype('float64')
        else:
            nums = pd.DataFrame()
        targ = df[self.targetcol]
        return pd.concat([cats, nums, targ], axis=1, sort=False)

    def _get_categories_list(self, df:pd.DataFrame):
        """
        get all possible values for categorical columns in the dataframe
        """
        categories = []
        for col in self.cat_cols:
            categories.append(df[col].unique())
        return categories

    def _get_categories_dict(self):
        """
        converts self.categories into a dict object
        """
        cat_vals = {}
        i = 0
        for cat in self.categories:
            cat_vals[self.cat_cols[i]] = cat
            i += 1
        return cat_vals

    def _add_miss_cols(self, df:pd.DataFrame, miss_chances:dict):
        """
        appends "_org" to all columns to be replaced with columns with nan misses
        """
        misser = MissTransformer(miss_chances=miss_chances)
        return misser.transform(df)

    def _encode_target_col(self, df:pd.DataFrame):
        """
        many a sklearn learners require target features to be numerical categories to work
        this converts the target columns labels into numerical values
        """
        df[self.targetcol] = LabelEncoder().fit_transform(df[self.targetcol])
        return df

    def _save_df(self, df:pd.DataFrame, summary_file:pd.DataFrame, 
                name:str, sub_folder:str):
        """
        saves a batch of data and updates the summary_file with a link of that data
        :param df: the data to be saved
        :param name: the name of the file
        :param folder: the folder into which the data batches will be saved in
        """
        if not os.path.exists(self.directory + sub_folder):
            os.makedirs(self.directory + sub_folder)
        pd.to_pickle(df, self.directory + sub_folder + '/' + name)
        #filepath = sub_folder + '/' + name
        filepath = name
        row = pd.DataFrame.from_dict(data={'filename':[filepath]})
        return summary_file.append(row, ignore_index=True)

    def _replace_with_missing(self, df:pd.DataFrame, index:int):
        """
        replaces miss column with original values up until index
        """
        for col in df:
            orgcol = col + '_org'
            if orgcol in df:
                df[col][:index] = df[orgcol][:index]
        return df

    def _create_batch_split(self, df:pd.DataFrame,
                        batch_size:int, ild_extra_rows:int,
                        sub_folder:str, summary_str:str, shuffle:bool):         
        """
        splits a dataframe into an initially labeled dataset with each class label
        at least represented once plus ild_extra_rows rows extra
        divides the rest of data into batch_size rows
        logs all created batch files in a summary file and serializes all data
        via pickle into files
        """
        summary = pd.DataFrame({'filename':[]})

        data = pd.DataFrame()
        
        if shuffle:
            #give each label a representation in ild          
            for category in df[self.targetcol].unique():
                shuffled = df.loc[lambda x: df[self.targetcol] == category, :].sample(frac=1)
                data = data.append(shuffled[:1])
                df.drop(index=shuffled.index[0], inplace=True)
            data = self._replace_with_missing(df=data, index=data.shape[0])
            
            #shuffle all data
            df = df.sample(frac=1)            
            
        df = self._replace_with_missing(df=df, index=ild_extra_rows)

        #add extra data to ild
        data = data.append(df[:ild_extra_rows])
        #check if first batch actually contains all labels
        if set(df[self.targetcol]) != set(data[self.targetcol]):
            raise ValueError("The initial data must contain all possible labels.")
        
        df.drop(df.index[:ild_extra_rows], inplace=True)
        summary = self._save_df(data, summary, 'labeled_set.pkl.gzip', sub_folder)

        #create batch files
        for i in range(0,df.shape[0],batch_size):
            summary = self._save_df(df[i:i+batch_size], 
                            summary, 'data{0}.pkl.gzip'.format(i),
                            sub_folder)

        summary.reset_index(inplace=True,drop=True)
        pd.to_pickle(summary, summary_str)

    def do_preprocessing(self, miss_chances:dict, batch_size:int, 
                    ild_extra_rows:int, sub_folder:str = "prepared", 
                    summary_str:str = None, shuffle:bool = True):
        """
        automatically converts a csv file with header row and ',' as separator
        into batches of pickled pandas.DataFrames with their data in specified
        columns in miss_chances altered to include misses of the likelihood
        specified in miss_chances
        the complete column is readded to the DataFrame under the original name + '_org'
        returns the categorical values of all categorical features
        """
        if summary_str is None: summary_str = self.summary_str()
        df = self.load_df()
        df = self._encode_target_col(df=df)
        pd.to_pickle(df, self.directory + 'raw_data.pkl.gzip')

        df = self._add_miss_cols(df=df, 
                        miss_chances=miss_chances)
        self._create_batch_split(df=df, batch_size=batch_size, 
                        ild_extra_rows=ild_extra_rows, 
                        sub_folder=sub_folder, 
                        summary_str=summary_str,
                        shuffle=shuffle)

    def do_framework(self, window, base_estimator, active_learner = None, 
                     oracle = None, feature_pipeline = None,
                     feature_acquisition = None, budget_manager = None,
                     summary_str:str = None,
                     pre_folder_name:str = "", post_folder_name:str = "",
                     evaluation_strategy = None, ild_timepoint = None, 
                     debug:bool=False):
        """
        does an active acquisition through the framework
        saves the summary.pkl.gzip also as tab separated csv file
        :param window: the window used for the stream
        :param budget_manager: the budget manager used by the AFA algorithm
        :param base_estimator: the final classifier to make predictions
        :param feature_pipeline: the pipeline used for the framework
        :param active_learner: the active learner to be used
        :param oracle: the oracle used in the framework
        :param feature_acquisition: the feature acquisition strategy used in the framework
        """
        # somehow directly starting a function through multiprocessing
        # removes window_data attribute from SlidingWindow despite being set to None
        # thus reset window_data back to None
        window.window_data = None
        if summary_str is None: summary_str = self.summary_str()
        if oracle is None:
            oracle = SimpleOracle()
        if active_learner is None:
            active_learner = NoActiveLearner(budget = 1.0, 
                                             oracle=oracle, 
                                             target_col_name=self.targetcol)
        if budget_manager is None:
            budget_manager = NoBudgetManager()
        if feature_acquisition is None:
            feature_acquisition = NoActiveFeatureAcquisition(budget_manager=budget_manager,
                                                            target_col_name=self.targetcol,
                                                            put_back=False,
                                                            acquisition_costs={},
                                                            debug=True)
        if feature_pipeline is None:
            feature_pipeline = self.get_default_pipeline()

        framework = FrameWork(
            summary_file=summary_str,
            base_estimator=base_estimator, 
            feature_pipeline=feature_pipeline,
            target_col_name=self.targetcol,
            ild_timepoint=ild_timepoint,
            feature_acquisition=feature_acquisition,
            active_learner=active_learner,
            window=window,
            evaluation_strategy=evaluation_strategy,
            results_path=self.directory,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            debug=debug)

        framework.process_data_stream()

        os.rename(os.path.join(framework.dir_result, framework.summary_filename), os.path.join(framework.dir_result, 'summary.pkl.gzip'))
        pd.read_pickle(os.path.join(framework.dir_result, 'summary.pkl.gzip')).to_csv(os.path.join(framework.dir_result, 'summary.csv'), sep="\t")

    def do_AFA_lower_bound(self, window, base_estimator, summary_str=None,
                           pre_folder_name="", post_folder_name=""):
        """
        does the lower bound calculation for the AFA task
        """
        print("Starting AFA lower bound on " + self.name)
        budget_manager = NoBudgetManager()
        feature_acquisition = NoActiveFeatureAcquisition(
                                target_col_name=self.targetcol, 
                                budget_manager=budget_manager)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager,
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
        
    def do_AFA_upper_bound(self, window, base_estimator, summary_str=None,
                           pre_folder_name="", post_folder_name=""):
        """
        does the upper bound calculation for the AFA task
        """
        print("Starting AFA upper bound on " + self.name)
        budget_manager = NoBudgetManager()
        feature_acquisition = RandomFeatureAcquisition(
                                target_col_name=self.targetcol, 
                                budget_manager=budget_manager,
                                put_back=True,
                                columns=self.cat_cols + self.num_cols)

        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager,
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)

    def do_AFA_random(self, window, base_estimator, budget_manager, put_back=False,
                      summary_str=None, pre_folder_name="", post_folder_name=""):
        """
        does an AFA task with random acquisition strategy
        """
        print("Starting random AFA on " + self.name)
        feature_acquisition = RandomFeatureAcquisition(
                                target_col_name=self.targetcol, 
                                budget_manager=budget_manager,
                                put_back=put_back,
                                columns=self.cat_cols+self.num_cols)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)

    def do_AFA_SAED(self, window, base_estimator, budget_manager, put_back=False,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SingleWindowAED(window=window,
                                target_col_name=self.targetcol, 
                                budget_manager=budget_manager,
                                acquisition_costs={},
                                categories=categories,
                                put_back=put_back,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)

    def do_AFA_MAED(self, window, base_estimator, budget_manager, aed_window_size, 
                    put_back=False, 
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a multi window
        average euclidean distance acquisition strategy
        """
        print("Starting MAED on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = MultiWindowAED(ild=window,
                                window_size=aed_window_size,
                                target_col_name=self.targetcol, 
                                budget_manager=budget_manager,
                                acquisition_costs={},
                                categories=categories,
                                put_back=put_back,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
    
    def do_AFA_SIG(self, window, base_estimator, budget_manager, put_back=False,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        information gain acquisition strategy
        """
        print("Starting SIG on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SingleWindowIG(window=window,
                                target_col_name=self.targetcol, 
                                budget_manager=budget_manager,
                                acquisition_costs={},
                                categories=categories,
                                put_back=put_back,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),#self.get_digitized_pipeline(bins),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager,
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)

    def do_AFA_SSU(self, window, base_estimator, budget_manager, put_back=False,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        symmetric uncertainty acquisition strategy
        """
        print("Starting SSU on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SingleWindowSU(window=window,
                                target_col_name=self.targetcol, 
                                budget_manager=budget_manager,
                                acquisition_costs={},
                                categories=categories,
                                put_back=put_back,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),#self.get_digitized_pipeline(bins),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)        
    
    def do_AFA_test(self, window, base_estimator, budget_manager, put_back=False,
                    summary_str=None, pre_folder_name="", post_folder_name=""):
        """
        various alternative AFA runs
        """
        print("Starting test AFA on " + self.name)
        feature_acquisition = osm.data_streams.active_feature_acquisition.supervised_merit_ranking.SingleWindowAED(window=window,
                        target_col_name=self.targetcol,
                        budget_manager=budget_manager,
                        acquisition_costs={},
                        categories=categories,
                        put_back=put_back,
                        debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
                     
def abalone():
    #4177 instances
    #8 features + 1 class
    #3 classes -> 
    return Dataset(
        name='abalone',
        directory=consts.DIR_CSV + '/abalone/',
        filename='abalone.csv',
        cat_cols=[],
        num_cols=['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                'Viscera weight', 'Shell weight', 'Rings'],
        pid_mins=[0, 0, 0, 0, 0, 0, 0, 0],
        pid_maxs=[1, 1, 1, 3, 2, 1, 1, 30],
        targetcol='Sex')
def adult():
    #32561 instances
    #13 features + 1 class
    #2 classes -> 
    return Dataset(
        name='adult',
        directory=consts.DIR_CSV + '/adult/',
        filename='adult.csv',
        cat_cols=['workclass', 'education', 'marital-status', 'occupation', 
                'relationship', 'race', 'sex', 'native-country'],
        num_cols=['age', 'capital-gain', 'capital-loss', 'hours-per-week'],
        pid_mins=[18, 0, 0, 0],
        pid_maxs=[100, 100000, 5000, 100],
        targetcol='label')
def airlines():
    #539383 instances
    #7 features + 1 class
    #2 classes ->
    return Dataset(
        name='airlines',
        directory=consts.DIR_CSV + '/airlines/',
        filename='airlines.csv',
        cat_cols=['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek'],
        num_cols=['Flight', 'Time', 'Length'],
        pid_mins=[1, 10, 23],
        pid_maxs=[7814, 1439, 655],
        targetcol='Delay')
def electricity():
    #45312 instances
    #8 features + 1 class
    #2 classes -> 
    return Dataset(
        name='electricity',
        directory=consts.DIR_CSV + '/electricity/',
        filename='elecNormNew.csv',
        cat_cols=['day'],
        num_cols=['date', 'period', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand',
        'transfer'],
        pid_mins=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        pid_maxs=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        targetcol='class')
def forest():
    #581012 instances
    #53 features + 1 class
    #7 classes -> 
    return Dataset(
        name='forest',
        directory=consts.DIR_CSV + '/forest/',
        filename='covtype.csv',
        cat_cols=['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
        'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
        'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
        'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
        'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
        'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',
        'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
        'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
        'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'],
        num_cols=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'],
        pid_mins=[2000, 0, 0, 0, -500, 0, 50, 50, 50, 0],
        pid_maxs=[3000, 360, 50, 1000, 500, 5000, 300, 300, 300, 7000],
        targetcol='class')
def intrusion():
    #494021 instances
    #41 features + 1 class
    #2 classes ->
    return Dataset(
        name='intrusion',
        directory=consts.DIR_CSV + '/intrusion/',
        filename='intrusion_10percent.csv',
        cat_cols=['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login',
        'is_guest_login'],
        num_cols=['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent',
        'hot', 'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'],
        pid_mins=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0],
        pid_maxs=[1000, 100000, 100000, 5, 10, 10, 10, 10, 10, 10,
                  10, 10, 10, 10, 10, 300, 300, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 300, 300, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0],
        targetcol='class')
def magic():
    #19020 instances
    #10 features + 1 class
    #2 classes -> 
    return Dataset(
        name='magic',
        directory=consts.DIR_CSV + '/magic_gamma_telescope/',
        filename='magic04.csv',
        cat_cols=[],
        num_cols=['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 
                'fM3Long', 'fM3Trans', 'fAlpha', 'fDist'],
        pid_mins=[5, 0, 2, 0, 0, -400, -300, -200, 0, 1],
        pid_maxs=[330, 250, 5, 1, 1, 500, 250, 200, 90, 500],
        targetcol='class')
def nursery():
    #12960 instances
    #8 features + 1 class
    #5 classes -> 
    return Dataset(
        name='nursery',
        directory=consts.DIR_CSV + '/nursery/',
        filename='nursery.csv',
        cat_cols=['parents', 'has_nurs', 'form', 'children', 'housing', 'finance',
                'social', 'health'],
        num_cols=[],
        pid_mins=[],
        pid_maxs=[],
        targetcol='label')
def occupancy():
    #20560 instances
    #6 features + 1 class
    #2 classes ->
    return Dataset(
        name='occupancy',
        directory=consts.DIR_CSV + '/occupancy/',
        filename='data.csv',
        cat_cols=['Weekday'],
        num_cols=['Hour', 'Minute', 'Temperature', 'Humidity', 'Light', 'CO2', 
                'HumidityRatio'],
        pid_mins=[0, 0, 15, 0, 0, 400, 0],
        pid_maxs=[23, 59, 35, 100, 2000, 2000, 1],
        targetcol='Occupancy')
def pendigits():
    #10992 instances
    #16 features + 1 class
    #10 classes -> 
    return Dataset(
        name='pendigits',
        directory=consts.DIR_CSV + '/pendigits/',
        filename='pendigits.csv',
        cat_cols=[],
        num_cols=['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'X5', 'Y5',
                'X6', 'Y6', 'X7', 'Y7', 'X8', 'Y8'],
        pid_mins=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        pid_maxs=[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        targetcol='Digit')
"""
def poker():
    #25010 instances
    #10 features + 1 class
    #10 classes -> 
    return Dataset(
        name='poker',
        directory=consts.DIR_CSV + '/poker_hand/',
        filename='poker-hand.csv',
        cat_cols=['S1', 'S2', 'S3', 'S4', 'S5'],
        num_cols=['C1', 'C2', 'C3', 'C4', 'C5'],
        targetcol='CLASS')
def popularity():
    #REGRESSION
    #39644 instances
    # features + 1 class
    #10 classes -> 
    raise NotImplementedError("Regression Tasks aren't possible yet")
    return Dataset(
        name='popularity',
        directory = consts.DIR_CSV + '/online_news_popularity/',
        filename = 'OnlineNewsPopularity.csv',
        cat_cols = [],
        num_cols = [],
        targetcol = 'shares')
"""
def sea():
    #60000 instances; snythetic
    # 3 features + 1 class
    # 2 classes -> [0, 1]; 4 concepts at 15k insts. each; fx + fy > [8, 9, 7, 9.5]
    return Dataset(
        name='sea',
        directory=consts.DIR_CSV + '/sea/',
        filename='sea.csv',
        cat_cols=[],
        num_cols=['f1', 'f2', 'f3'],
        pid_mins=[0, 0, 0],
        pid_maxs=[10, 10, 10],
        targetcol='label')
def generator(filename):
    #??? instances
    #?? features + 1 class
    #2 classes -> 0, 1
    directory = os.path.join(consts.DIR_GEN, filename) + '\\'
    f = open(os.path.join(directory, "params.txt"), "r")
    features = 0
    for line in f:
        if "features" in line:
            features = int(line.replace("features\t", ""))#'ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz\t\r\n'))
            break
    f.close()

    return Dataset(
        name='gen',
        directory=directory, #+ '/gen_or_25_3_True_50_20_10/',
        filename='raw_data.csv',
        cat_cols=list(map(str, range(features))),
        num_cols=[],
        pid_mins=[],
        pid_maxs=[],
        targetcol='label')
def get_data_set(name:str):
    """
    returns the dataset corresponding to its name
    returns None if no match found
    """
    if name == 'abalone': return abalone()
    elif name == 'adult': return adult()
    elif name == 'airlines': return airlines()
    elif name == 'electricity': return electricity()
    elif name == 'forest': return forest()
    elif name == 'intrusion': return intrusion()
    elif name == 'magic': return magic()
    elif name == 'nursery': return nursery()
    elif name == 'occupancy': return occupancy()
    elif name == 'pendigits': return pendigits()
    elif name == 'sea': return sea()
    elif 'gen' in 'name': return generator(name)        
      
def title(text:str):
    os.system("title " + text)

def get_slwin(window_size:int, forgetting_strategy=None):
    """
    returns a new sliding window
    """
    return SlidingWindow(window_size=window_size, forgetting_strategy=forgetting_strategy)

def get_flwin(window_size:int):
    """
    returns a new fixed length window
    """
    return FixedLengthWindow(window_size=window_size)
    
def get_ipf(budget:float, window_size:int):
    """
    returns a new incremental percentile filter budget manager
    """
    return IncrementalPercentileFilter(budget_threshold=budget, window_size=window_size)

def get_sbm(budget:float):
    """
    returns a new simple budget manager
    """
    return SimpleBudgetManager(budget_threshold=budget)

def calc_ident_miss_chances(dataset:Dataset, miss_chance:float):
    miss_chances = {}
    for col in dataset.num_cols + dataset.cat_cols:
        miss_chances[col] = miss_chance
    return miss_chances

    
def _mp_do_all_task(dataset, aed_window_size, window_size, ipf_size, batch_size, ild_extra_rows, budgets, miss_chance, i):
    """
    called from do_tasks_all_afas_bms as subprocess
    thus freeing memory after completion as python refuses to free memory in free lists
    """
    sub_folder = 'prepared/' + str(i) + '/' + str(miss_chance)
    summary_str = dataset.summary_str('prepared', str(i), str(miss_chance))
    pre_folder_name=str(i)
    post_folder_name=str(miss_chance)
    """
    #preprocessing
    title(str(i) + "-th Preprocessing " + str(miss_chance) + " misses " + dataset.name)
    dataset.do_preprocessing(miss_chances=calc_ident_miss_chances(dataset, miss_chance),
                        shuffle=False,
                        batch_size=batch_size, ild_extra_rows=ild_extra_rows,
                        summary_str=summary_str,
                        sub_folder=sub_folder)
    #""
    #lower bound
    title(str(i) + "-th Lower Bound " + str(miss_chance) + " misses " + dataset.name)
    window = get_slwin(window_size)
    base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)#SGDPredictFix()#CalibratedClassifierCV(SGDClassifier(max_iter=100, tol=1e-3), cv=3)
    dataset.do_AFA_lower_bound(window=window, 
                            base_estimator=base_estimator,
                            summary_str=summary_str,
                            pre_folder_name=pre_folder_name,
                            post_folder_name=post_folder_name)
    #""
    #upper bound
    title(str(i) + "-th Upper Bound " + str(miss_chance) + " misses " + dataset.name)
    window = get_slwin(window_size)
    base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
    dataset.do_AFA_upper_bound(window=window, 
                            base_estimator=base_estimator,
                            summary_str=summary_str,
                            pre_folder_name=pre_folder_name,
                            post_folder_name=post_folder_name)
    """                       
    for budget in budgets:
        #for skip_quality in [True, False]:
        print(dataset.name.upper() + ' ' + str(budget))
        
        #""
        #random AFA
        title(str(i) + "-th Random AFA + SBM " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_sbm(budget)
        dataset.do_AFA_random(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
        #""
        #SAED
        #""
        title(str(i) + "-th SAED + IPF " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_ipf(budget, ipf_size)
        dataset.do_AFA_SAED(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
                        #post_folder_name=post_folder_name + ' ' + str(skip_quality),
                        #debug=skip_quality)
        #""
        title(str(i) + "-th SAED + SBM " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_sbm(budget)
        dataset.do_AFA_SAED(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)        
        #""        
        #SIG
        #""
        title(str(i) + "-th SIG + IPF " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_ipf(budget, ipf_size)
        dataset.do_AFA_SIG(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
        #""                
        title(str(i) + "-th SIG + SBM " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_sbm(budget)
        dataset.do_AFA_SIG(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
                        
        #SSU
        #""
        title(str(i) + "-th SSU + IPF " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_ipf(budget, ipf_size)
        dataset.do_AFA_SSU(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
        #""                
        title(str(i) + "-th SSU + SBM " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_sbm(budget)
        dataset.do_AFA_SSU(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
        #""
        #MAED
        title(str(i) + "-th MAED + IPF " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)# + str(skip_quality))
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_ipf(budget, ipf_size)
        dataset.do_AFA_MAED(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        aed_window_size=aed_window_size,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
                        #post_folder_name=post_folder_name + ' ' + str(skip_quality),
                        #debug=skip_quality)
        #""
        title(str(i) + "-th MAED + SBM " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_sbm(budget)
        dataset.do_AFA_MAED(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        aed_window_size=aed_window_size,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
        #""
        
def do_tasks_all_afas_bms(datasets:list = [abalone()],
                       aed_window_size:int = 50, window_size:int = 10,
                       ipf_size:int = 50, batch_size:int = 50, ild_extra_rows:int = 50,
                       budgets:list = [0.5], miss_chances:list = [0.5], iterations = [0]):
    """
    Executes a list of datasets with the given tasks with all given Missignesses and Budgets
    combinations over the specified dataset iterations times
    """
    for dataset in datasets:
        for i in iterations:#range(iterations):
            #pool = mp.Pool(processes=4)
            #pool_args = []
            for miss_chance in miss_chances:
                #pool_args.append([dataset, aed_window_size, window_size,
                #    ipf_size, batch_size, ild_extra_rows, budgets, miss_chance, i])
            #pool.starmap(_mp_do_all_task, pool_args)
                p = mp.Process(target=_mp_do_all_task, 
                    args=(dataset, aed_window_size, window_size,
                    ipf_size, batch_size, ild_extra_rows, budgets, miss_chance, i))
                p.start()
                p.join()

def do_tasks(params:dict):
    """
    Starts a new process for each param
    :param params: Dictionary with parameters
    """
    # fail cases
    window_size = params['w']
    datasets = params['d']
    iterations = params['i']
    miss_chances = params['m']
    budgets = params['b']
    tasks = params['t']
    preparation = params['p']
    log_filename = params['l']
    ipf_size = 50
    aed_size = 50
    if not window_size: raise ValueError("Window size has to be specified")
    if len(datasets) == 0: raise ValueError("At least one dataset has to be specified")
    if len(iterations) == 0: raise ValueError("An iteration value has to be specified")
    
    # prep, lower and upper
    prepare = len(preparation) != 0
    if prepare: shuffle, batch_size, ild_size = preparation
    lower = 'lower' in tasks
    if lower: tasks.remove('lower')
    upper = 'upper' in tasks
    if upper: tasks.remove('upper')
    
    log = open(log_filename, "w")
    log.write("params:" + str(params) + '\n')
    
    for dataset in datasets:
        for iteration in iterations:
            for miss_chance in miss_chances:
                sub_folder = 'prepared/' + str(iteration) + '/' + str(miss_chance)
                summary_str = dataset.summary_str('prepared', str(iteration), str(miss_chance))
                pre_folder_name = str(iteration)
                post_folder_name = str(miss_chance)
                
                #preprocessing
                if prepare:
                    text = "{} {}-th preprocessing {} misses".format(dataset, iteration, miss_chance)
                    title(text)
                    log.write("{}: {}\n".format(datetime.now(), text))
                    p = mp.Process(target=dataset.do_preprocessing, 
                        args=(calc_ident_miss_chances(dataset, miss_chance), 
                            batch_size, ild_size, sub_folder, summary_str, shuffle))
                    p.start()
                    p.join()
                    
                #lower
                if lower:
                    text = "{} {}-th lower bound {} misses".format(dataset, iteration, miss_chance)
                    title(text)
                    log.write("{}: {}\n".format(datetime.now(), text))
                    p = mp.Process(target=dataset.do_AFA_lower_bound,
                        args=(get_slwin(window_size), SGDClassifier(loss='log', max_iter=100, tol=1e-3),
                            summary_str, pre_folder_name, post_folder_name))
                    p.start()
                    p.join()
                
                #upper
                if upper:
                    text = "{} {}-th upper bound {} misses".format(dataset, iteration, miss_chance)
                    title(text)
                    log.write("{}: {}\n".format(datetime.now(), text))
                    p = mp.Process(target=dataset.do_AFA_upper_bound,
                        args=(get_slwin(window_size), SGDClassifier(loss='log', max_iter=100, tol=1e-3),
                            summary_str, pre_folder_name, post_folder_name))
                    p.start()
                    p.join()
                
                for budget in budgets:
                    for task in tasks:
                        #task
                        text = "{} {}-th {} {} misses {} budget".format(dataset, iteration, task, miss_chance, budget)
                        title(text)
                        log.write("{}: {}\n".format(datetime.now(), text))
                        afa_s, bm_s = task.split('+')
                        
                        bm = None
                        if bm_s == "IPF": bm = IncrementalPercentileFilter(budget_threshold=budget, window_size=ipf_size)
                        elif bm_s == "SBM": bm = SimpleBudgetManager(budget_threshold=budget)
                        elif bm_s == "NBM": bm = NoBudgetManager()
                        
                        if afa_s == "RA": p = mp.Process(target=dataset.do_AFA_random,
                            args=(get_slwin(window_size), SGDClassifier(loss='log', max_iter=100, tol=1e-3),
                                bm, False, summary_str, pre_folder_name, post_folder_name))
                        elif afa_s == "SWAED": p = mp.Process(target=dataset.do_AFA_SAED,
                            args=(get_slwin(window_size), SGDClassifier(loss='log', max_iter=100, tol=1e-3),
                                bm, False, summary_str, pre_folder_name, post_folder_name, True))
                        elif afa_s == "SWIG": p = mp.Process(target=dataset.do_AFA_SIG,
                            args=(get_slwin(window_size), SGDClassifier(loss='log', max_iter=100, tol=1e-3),
                                bm, False, summary_str, pre_folder_name, post_folder_name, True))
                        elif afa_s == "SWSU": p = mp.Process(target=dataset.do_AFA_SSU,
                            args=(get_slwin(window_size), SGDClassifier(loss='log', max_iter=100, tol=1e-3),
                                bm, False, summary_str, pre_folder_name, post_folder_name, True))
                        elif afa_s == "MWAED": p = mp.Process(target=dataset.do_AFA_MAED,
                            args=(get_slwin(window_size), SGDClassifier(loss='log', max_iter=100, tol=1e-3),
                                bm, aed_size, False, summary_str, pre_folder_name, post_folder_name, True))
                        p.start()
                        p.join()
    log.close()

def remove_chars(text:str):
    return text.translate({ord(i): None for i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,:;()[]{}+*-#~_ '})

def abbreviate(text:str, delim:str = '_'):
    retval = ''
    for word in text.split(delim):
        retval += word[0].upper()
    return retval

def valid_afa_bm(text:str):
    """
    Returns whether the AFA and BM are contained in constants
    """
    if text == 'upper': return True
    if text == 'lower': return True
    afa, bm = text.split('+')
    return afa in list(map(abbreviate, consts.AFAS)) and bm in list(map(abbreviate, consts.BMS))

if __name__ == '__main__':
    """
    executes runs given hardcoded parameters
    contains dataset informations and Dataset class
    
    the following space separated args are defined:
    -d                          - checks following args for defined dataset names
    after -d [dataset]          - adds dataset to the tasks
    -t                          - checks following args for defined afa+bm combinations
    after -t [task or afa+bm]   - 
    -i                          - 
    after -i [iterations]       - adds specific iterations to the process, accepts val..val as ranges
    -b                          - checks following args for budget values
    after -b                    -
    -m                          - checks following args for missingness values
    after -m                    -
    -p                          - enables the preprocessing step for all data sets
    after -p [shuffle] [batch size] [additional ild instances]
                                - whether to randomize the order of the data set 
                                    if shuffled, adds one representative instance for each label
                                - how large the batch size is
                                - the amount of additional instances in the initial data
    -l                          - checks the following args for the name of a log file
    after -l [log filename]     - the name the log writes into
    -w                          - checks the following args for the window size
    after -w [window size]      - the size (in batches) of the sliding window
    
    use example:
        -d nursery -p True 50 100 -t lower upper RA+SBM SWAED+SBM SWAED+IPF -i 0..3 9 -m 0.25 0.5 0.75 -b 0.25 0.5 0.75 1.0 -w 10 -l tasklogfile.log
    """
    datasets = []
    #budgets = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    budgets = []
    #budgets = [1.0]
    #miss_chances = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    miss_chances = []
    #miss_chances = [0.75]
    iterations = []
    tasks = []
    preparation = []
    
    params = {'d':datasets, 't':tasks, 'i':iterations, 'b':budgets, 'm':miss_chances, 'p':preparation, 'l':"tasks.log", 'w':None}
    
    read_mode = ''
    for arg in sys.argv[1:]:        
        if arg[0] == '-' and arg[1] in params: read_mode = arg[1]
        else:
            if read_mode == 'l': params['l'] = arg
            elif read_mode == 'd':
                dataset = get_data_set(arg)
                if dataset: datasets.append(dataset)
            elif read_mode == 't':
                if valid_afa_bm(arg): tasks.append(arg)
            elif read_mode == 'p':
                if arg in ['T', 't', 'True', 'true']: preparation.append(True)
                elif arg in ['F', 'f', 'False', 'false']: preparation.append(False)
                else: preparation.append(int(remove_chars(arg)))
            elif read_mode == 'w': params[read_mode] = int(remove_chars(arg))                
            else:
                if '..' in arg:
                    args = arg.split('..')
                    params[read_mode] += range(int(args[0]), int(args[1]))
                else: params[read_mode].append(float(remove_chars(arg)))
        
    print("data sets   : {}".format(datasets))
    print("iterations  : {}".format(iterations))
    print("miss_chances: {}".format(miss_chances))
    print("budgets     : {}".format(budgets))
    if len(preparation) != 0: print("preparation : shuffle {}, batch size {}, ild extra instances {}".format(preparation[0], preparation[1], preparation[2]))
    print("window size : {}".format(params['w']))
    print("tasks       : {}".format(tasks))
    print("log file    : {}".format(params['l']))
    print("")
    c = input("To continue enter \'y\'\n\n")
    if c == 'y': do_tasks(params)
    else: do_tasks_all_afas_bms(datasets=[abalone()],
                       aed_window_size=50, window_size=10,
                       ipf_size=50, batch_size=50, ild_extra_rows=50,
                       budgets=[0.125], miss_chances=[0.125], iterations=[10])