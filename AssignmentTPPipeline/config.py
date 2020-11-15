# ====   PATHS ===================

TRAINING_DATA_FILE = "titanic.csv"
PIPELINE_NAME = "logistic_regression.pkl"


# ======= FEATURE GROUPS =============

TARGET = "survived"

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_VARS_TO_IMPUTE = ['age', 'fare']

NUMERICAL_VARS = ['pclass', 'age', 'sibsp', 'parch', 'fare']

CABIN = 'cabin'

FEATURES = NUMERICAL_VARS + CATEGORICAL_VARS