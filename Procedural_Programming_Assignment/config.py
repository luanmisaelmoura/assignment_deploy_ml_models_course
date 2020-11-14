# ====   PATHS ===================

PATH_TO_DATASET = "titanic.csv"
OUTPUT_SCALER_PATH = 'scaler.pkl'
ENCODER_PATH = 'encoder.pkl'
OUTPUT_MODEL_PATH = 'logistic_regression.pkl'


# ======= PARAMETERS ===============

# imputation parameters
IMPUTATION_DICT = \
    {'age': 28.0,
    'fare': 14.4542,
    'sex': 'Missing',
    'cabin': 'Missing',
    'embarked': 'Missing',
    'title': 'Missing'}


# encoding parameters
FREQUENT_LABELS = \
    {'sex': ['female', 'male'],
    'cabin': ['C', 'Missing'],
    'embarked': ['C', 'Q', 'S'],
    'title': ['Miss', 'Mr', 'Mrs']}

DUMMY_VARIABLES = \
    ['sex_male', 'cabin_Missing', 'cabin_Rare',
    'embarked_S', 'embarked_C', 'embarked_Q',
    'title_Mr', 'title_Miss', 'title_Mrs']


# ======= FEATURE GROUPS =============

TARGET = 'survived'

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_TO_IMPUTE = ['age', 'fare']