from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import config


titanic_pipe = Pipeline(
    # complete with the list of steps from the preprocessors file
    # and the list of variables from the config
    [   
        ('extract_first_letter',
            pp.ExtractFirstLetter(variables=config.CABIN)),
        
        ('missing_indicator',
            pp.MissingIndicator(variables=config.NUMERICAL_VARS_TO_IMPUTE)),

        ('numerical_imputer',
            pp.NumericalImputer(variables=config.NUMERICAL_VARS_TO_IMPUTE)),

        ('categorical_imputer',
            pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)),
        
        ('rare_label_categorical_encoder',
            pp.RareLabelCategoricalEncoder(variables=config.CATEGORICAL_VARS)),
        
        ('categorical_encoder',
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),

        ('standard_scaler',
            StandardScaler()),

        ('logistic_regression',
            LogisticRegression(C=1/0.0005, random_state=0))
    ]
)
