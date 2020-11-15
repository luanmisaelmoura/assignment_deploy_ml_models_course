import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.categorical_encoders import OneHotCategoricalEncoder

# Add binary variable to indicate missing values
class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables =  variables

    def fit(self, X, y=None):
        # to accommodate sklearn pipeline functionality
        return self


    def transform(self, X):
        # add indicator
        X = X.copy()
        
        for feature in self.variables:
            X[feature+'_NA'] = \
                np.where(X[feature].isna(), 1, 0)
        
        return X


# categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            X[feature].fillna('Missing', inplace=True)
        
        return X


# Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist median values in a dictionary
        self.imputer_dict_ = dict()

        for feature in self.variables:
            self.imputer_dict_[feature] = \
                X[feature].median()

        return self

    def transform(self, X):

        X = X.copy()

        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)

        return X


# Extract first letter from string variable
class ExtractFirstLetter(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        
        for feature in self.variables:
            X[feature] = X[feature].str[0]

        return X


# frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):
        
        self.tol = tol

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):

        # persist frequent labels in dictionary
        self.encoder_dict_ = dict()
        
        X = X.copy()

        for feature in self.variables:
            tmp = \
                X.loc[:, feature]\
                    .value_counts(normalize=True)
            
            self.encoder_dict_[feature] = \
                tmp[tmp > self.tol].index

        return self

    def transform(self, X):
        
        X = X.copy()

        for feature in self.variables:
            X[feature] = \
                np.where(
                    X[feature].isin(
                        self.encoder_dict_[feature]),
                    X[feature],
                    'Rare')

        return X


# string to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    # followed the link creating a simpler version:
    # https://github.com/solegalli/feature_engine/blob/master/feature_engine/encoding/one_hot.py
    def __init__(self, variables=None, drop_last=False):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.drop_last = drop_last

    def fit(self, X, y=None):

        self.encoder_dict_ = dict()

        for feature in self.variables:
            category_ls = X[feature].unique().tolist()
            if self.drop_last:
                self.encoder_dict_[feature] = \
                    category_ls[:-1]
            else:
                self.encoder_dict_[feature] = \
                    category_ls
        
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        
        for feature in self.variables:
            for category in self.encoder_dict_[feature]:
                X[feature + '_' + category] = \
                    np.where(X[feature] == category, 1, 0)

        X.drop(columns=self.variables, inplace=True)

        return X
