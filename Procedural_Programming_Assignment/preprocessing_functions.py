import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from feature_engine.categorical_encoders import OneHotCategoricalEncoder
import joblib
import config


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    return pd.read_csv(df_path)


def divide_train_test(df, target):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = \
        train_test_split(
        df.drop(target, axis=1),  # predictors
        df[target],  # target
        test_size=0.2,  # percentage of obs in test set
        random_state=0)  # seed to ensure reproducibility
    return X_train, X_test, y_train, y_test


def extract_cabin_letter(df, var):
    # captures the first letter
    df[var] = df[var].str[0]


def add_missing_indicator(df, var):
    # function adds a binary missing value indicator
    df[var+'_NA'] = \
        np.where(df[var].isnull(), 1, 0)


def impute_na(df, var, value):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    df[var].fillna(value, inplace=True)


def remove_rare_labels(df, var, frequent_ls):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    df[var] = \
        np.where(df[var].isin(frequent_ls), df[var], 'Rare')

def fit_cat_encoder(df, vars_cat, output_path):   
    encoder = OneHotCategoricalEncoder(
    top_categories=None,
    variables=vars_cat,
    drop_last=True)

    encoder.fit(df)
    joblib.dump(encoder, output_path)
    

def encode_categorical(df, output_path):
    encoder = \
        joblib.load(output_path)
    # adds ohe variables and removes original categorical variable
    return encoder.transform(df)


def check_dummy_variables(df, dummy_list):
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    assert np.isin(dummy_list, df.columns).all()

def train_scaler(df, output_path):
    # train and save scaler
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)


def scale_features(df, output_path):
    # load scaler and transform data
    scaler = joblib.load(output_path)
    df_columns = df.columns
    df = scaler.transform(df)
    df = pd.DataFrame(df, columns = df_columns)


def train_model(df, target, output_path):
    # train and save model
    model = \
        LogisticRegression(C=1/0.0005, random_state=0)
    model.fit(df, target)
    joblib.dump(model, output_path)


def predict(df, output_path):
    model = joblib.load(output_path)
    predictions = model.predict(df)
    return predictions