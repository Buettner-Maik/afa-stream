# plotting consts
AFAS = ['multi_window_average_euclidean_distance', 'single_window_average_euclidean_distance',
            'random_acquisition', 'no_active_feature_acquisition', 'single_window_information_gain',
            'single_window_symmetric_uncertainty']
MWAED = AFAS[0]
SWAED = AFAS[1]
RA = AFAS[2]
NAFA = AFAS[3]
SWIG = AFAS[4]
SWSU = AFAS[5]            
            
            
BMS = ['incremental_percentile_filter', 'simple_budget_manager', 'no_budget_manager']
IPF = BMS[0]
SBM = BMS[1]
NBM = BMS[2]

BUDGETS = ['0.125', '0.25', '0.375', '0.5', '0.625', '0.75', '0.875', '1.0']
MCS = ['0.125', '0.25', '0.375', '0.5', '0.625', '0.75', '0.875']
DATASETS = ['abalone', 'adult', 'electricity', 'gen', 'magic', 'nursery', 'occupancy', 'pendigits', 'sea']
CLASSIFIERS = ['sgd', 'dtc']
KEEP_NORMS = [True, False]
WINDOW_SIZES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# extensions
EXT_ARFF = ".arff"

# file paths
DIR_ARFF = "data/arff"
DIR_CSV = "data/csv"
DIR_GEN = "data/gen"
DIR_RPG = "data/arff/rpg"
DIR_PLOTS = "data/plots/auto"

#MISS_FEATURE = float('NaN')