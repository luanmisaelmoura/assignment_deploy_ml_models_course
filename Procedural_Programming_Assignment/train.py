import preprocessing_functions as pf
import config

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
df = pf.load_data(config.PATH_TO_DATASET)

# divide data set
X_train, X_test, y_train, y_test = \
    pf.divide_train_test(df=df, target='survived')

# get first letter from cabin variable
pf.extract_cabin_letter(X_train, 'cabin')
pf.extract_cabin_letter(X_test, 'cabin')

# impute categorical variables
for var in config.CATEGORICAL_VARS:
    pf.impute_na(X_train, var, config.IMPUTATION_DICT[var])
    pf.impute_na(X_test, var, config.IMPUTATION_DICT[var])

# impute numerical variable
pf.add_missing_indicator(X_train, 'age')
pf.add_missing_indicator(X_test, 'age')
pf.add_missing_indicator(X_train, 'fare')
pf.add_missing_indicator(X_test, 'fare')

for var in config.NUMERICAL_TO_IMPUTE:
    pf.impute_na(X_train, var, config.IMPUTATION_DICT[var])
    pf.impute_na(X_test, var, config.IMPUTATION_DICT[var])

# Group rare labels
for var in config.CATEGORICAL_VARS:
    pf.remove_rare_labels(X_train, var, config.FREQUENT_LABELS[var])
    pf.remove_rare_labels(X_test, var, config.FREQUENT_LABELS[var])

pf.fit_cat_encoder(X_train, config.CATEGORICAL_VARS, config.ENCODER_PATH)
# encode categorical variables
X_train = pf.encode_categorical(X_train, config.ENCODER_PATH)
X_test = pf.encode_categorical(X_test, config.ENCODER_PATH)

# check all dummies were added
pf.check_dummy_variables(X_train, config.DUMMY_VARIABLES)
pf.check_dummy_variables(X_test, config.DUMMY_VARIABLES)

# train scaler and save
pf.train_scaler(X_train, config.OUTPUT_SCALER_PATH)

# scale train set
pf.scale_features(X_train, config.OUTPUT_SCALER_PATH)
pf.scale_features(X_test, config.OUTPUT_SCALER_PATH)

# train model and save
pf.train_model(X_train, y_train, config.OUTPUT_MODEL_PATH)

print('Finished training')