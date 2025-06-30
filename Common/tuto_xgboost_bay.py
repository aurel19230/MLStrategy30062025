# Dataset
from sklearn import datasets

# Data processing
import pandas as pd
import numpy as np

# Standardize the data
from sklearn.preprocessing import StandardScaler

# Model and performance evaluation
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support as score

# Hyperparameter tuning
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval

# Load the breast cancer dataset
data = datasets.load_breast_cancer()

# Put the data in pandas dataframe format
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target']=data.target

# Check the data information
df.info()

# Check the target value distribution
df['target'].value_counts(normalize=True)
# Train test split
X_train, X_test, y_train, y_test = train_test_split(df[df.columns.difference(['target'])],
                                                    df['target'],
                                                    test_size=0.2,
                                                    random_state=42)

# Check the number of records in training and testing dataset.
print(f'The training dataset has {len(X_train)} records.')
print(f'The testing dataset has {len(X_test)} records.')

# Initiate scaler
sc = StandardScaler()

# Standardize the training dataset
X_train_transformed = pd.DataFrame(sc.fit_transform(X_train),index=X_train.index, columns=X_train.columns)

# Standardized the testing dataset
X_test_transformed = pd.DataFrame(sc.transform(X_test),index=X_test.index, columns=X_test.columns)

# Summary statistics after standardization
X_train_transformed.describe().T

# Summary statistics before standardization
X_train.describe().T
print(X_train_transformed)

# Initiate XGBoost Classifier
xgboost = XGBClassifier()

# Print default setting
xgboost.get_params()

# Train the model
xgboost = XGBClassifier(seed=0).fit(X_train_transformed,y_train)

# Make prediction
xgboost_predict = xgboost.predict(X_test_transformed)

# Get predicted probability
xgboost_predict_prob = xgboost.predict_proba(X_test_transformed)[:,1]
# Train the model
xgboost = XGBClassifier(seed=0).fit(X_train_transformed,y_train)

# Make prediction
xgboost_predict = xgboost.predict(X_test_transformed)

# Get predicted probability
xgboost_predict_prob = xgboost.predict_proba(X_test_transformed)[:,1]
# Get performance metrics
precision, recall, fscore, support = score(y_test, xgboost_predict)

# Print result
print(f'The recall value for the baseline xgboost model is {recall[1]:.4f}')
# Print result
print(f'The precision value for the baseline xgboost model is {precision[1]:.4f}')

# Define the search space
param_grid = {
    # Percentage of columns to be randomly samples for each tree.
    "colsample_bytree": [ 0.3, 0.5 , 0.8 ],
    # reg_alpha provides l1 regularization to the weight, higher values result in more conservative models
    "reg_alpha": [0, 0.5, 1, 5],
    # reg_lambda provides l2 regularization to the weight, higher values result in more conservative models
    "reg_lambda": [0, 0.5, 1, 5]
    }

# Set up score
scoring = ['recall']

# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
# Define the search space

# Set up score
scoring = ['recall']

# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
# Define grid search
grid_search = GridSearchCV(estimator=xgboost,
                           param_grid=param_grid,
                           scoring=scoring,
                           refit='recall',
                           n_jobs=-1,
                           cv=kfold,
                           verbose=0)

# Fit grid search
grid_result = grid_search.fit(X_train_transformed, y_train)

# Print grid search summary
print(grid_result)

# Print the best score and the corresponding hyperparameters
print(f'The best score is {grid_result.best_score_:.4f}')
print('The best score standard deviation is', round(grid_result.cv_results_['std_test_recall'][grid_result.best_index_], 4))
print(f'The best hyperparameters are {grid_result.best_params_}')

# Make prediction using the best model
grid_predict = grid_search.predict(X_test_transformed)

# Get predicted probabilities
grid_predict_prob = grid_search.predict_proba(X_test_transformed)[:,1]

# Get performance metrics
precision, recall, fscore, support = score(y_test, grid_predict)

# Print result
print(f'The recall value for the xgboost grid search is {recall[1]:.4f}')

# Define the search space
param_grid = {
    # Learning rate shrinks the weights to make the boosting process more conservative
    "learning_rate": [0.0001,0.001, 0.01, 0.1, 1] ,
    # Maximum depth of the tree, increasing it increases the model complexity.
    "max_depth": range(3,21,3),
    # Gamma specifies the minimum loss reduction required to make a split.
    "gamma": [i/10.0 for i in range(0,5)],
    # Percentage of columns to be randomly samples for each tree.
    "colsample_bytree": [i/10.0 for i in range(3,10)],
    # reg_alpha provides l1 regularization to the weight, higher values result in more conservative models
    "reg_alpha": [1e-5, 1e-2, 0.1, 1, 10, 100],
    # reg_lambda provides l2 regularization to the weight, higher values result in more conservative models
    "reg_lambda": [1e-5, 1e-2, 0.1, 1, 10, 100]}

# Set up score
scoring = ['recall']

# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

# Define random search
random_search = RandomizedSearchCV(estimator=xgboost,
                           param_distributions=param_grid,
                           n_iter=10,
                           scoring=scoring,
                           refit='recall',
                           n_jobs=-1,
                           cv=kfold,
                           verbose=0)

# Fit grid search
random_result = random_search.fit(X_train_transformed, y_train)

# Print grid search summary
random_result

# Print the best score and the corresponding hyperparameters
print(f'The best score is {random_result.best_score_:.4f}')
print('The best score standard deviation is', round(random_result.cv_results_['std_test_recall'][random_result.best_index_], 4))
print(f'The best hyperparameters are {random_result.best_params_}')

# Space
print("\n------------------Bayesian Optimization For XGBoost-------------------------------\n")
space = {
    'learning_rate': hp.choice('learning_rate', [0.0001,0.001, 0.01, 0.1, 1]),
    'max_depth' : hp.choice('max_depth', range(3,21,3)),
    'gamma' : hp.choice('gamma', [i/10.0 for i in range(0,5)]),
    'colsample_bytree' : hp.choice('colsample_bytree', [i/10.0 for i in range(3,10)]),
    'reg_alpha' : hp.choice('reg_alpha', [1e-5, 1e-2, 0.1, 1, 10, 100]),
    'reg_lambda' : hp.choice('reg_lambda', [1e-5, 1e-2, 0.1, 1, 10, 100])
}

# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


# Objective function
def objective(params):
    xgboost = XGBClassifier(seed=0, **params)
    score = cross_val_score(estimator=xgboost,
                            X=X_train_transformed,
                            y=y_train,
                            cv=kfold,
                            scoring='recall',
                            n_jobs=-1).mean()

    # Loss is negative score
    loss = - score

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = 48, trials = Trials())

# Print the index of the best parameters
print(best)

# Print the values of the best parameters
print(space_eval(space, best))

# Train model using the best parameters
xgboost_bo = XGBClassifier(seed=0,
                           colsample_bytree=space_eval(space, best)['colsample_bytree'],
                           gamma=space_eval(space, best)['gamma'],
                           learning_rate=space_eval(space, best)['learning_rate'],
                           max_depth=space_eval(space, best)['max_depth'],
                           reg_alpha=space_eval(space, best)['reg_alpha'],
                           reg_lambda=space_eval(space, best)['reg_lambda']
                           ).fit(X_train_transformed,y_train)

# Make prediction using the best model
bayesian_opt_predict = xgboost_bo.predict(X_test_transformed)

# Get predicted probabilities
bayesian_opt_predict_prob = xgboost_bo.predict_proba(X_test_transformed)[:,1]

# Get performance metrics
precision, recall, fscore, support = score(y_test, bayesian_opt_predict)

# Print result
print(f'The recall value for the xgboost Bayesian optimization is {recall[1]:.4f}')