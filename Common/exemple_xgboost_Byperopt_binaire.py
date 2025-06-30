# import pandas for data wrangling
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
from standardFunc_sauv import plot_feature_histograms_by_class
import matplotlib.pyplot as plt

# import numpy for Scientific computations
import numpy as np

# import machine learning libraries
import xgboost as xgb
from sklearn.metrics import accuracy_score

# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# import packages for feature importance
from sklearn.inspection import permutation_importance
import shap


# import seaborn for correlation heatmap
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#data = '/kaggle/input/wholesale-customers-data-set/Wholesale customers data.csv'
file_path = "C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/Sierra chart/xTickReversal/data_simuNew/example/Wholesale customers data.csv"

df = pd.read_csv(file_path)

X = df.drop('Channel', axis=1)

y = df['Channel']
X.head()
y.head()
# convert labels into binary values

y[y == 2] = 0

y[y == 1] = 1

df_concat = pd.concat([X, y], axis=1)

print("DataFrame concaténé:")
print(df_concat.head())


column_settings = {
    'Region':    (False, False, 10, 90),
    'Fresh':    (False, False, 10, 90),
    'Milk':   (True, True, 10, 90),
    'Grocery':    (True, True, 10, 90),
    'Frozen':      (True, True, 10, 90),
    'Detergents_Paper':     (True, True, 10, 90),
    'Delicassen':      (True, True, 10, 90)
}

plot_feature_histograms_by_class(df_concat, 'Channel',column_settings, figsize=(32, 24))





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
space = {'max_depth': hp.quniform("max_depth", 3, 20, 1),
         'gamma': hp.uniform('gamma', 1, 10),
         'reg_alpha': hp.quniform('reg_alpha', 1, 200, 1),
         'reg_lambda': hp.uniform('reg_lambda', 0, 1),
         'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
         'min_child_weight': hp.quniform('min_child_weight', 0, 20, 1),
         'n_estimators': 180,
         'eta': 0.1,
         'seed': 0
         }


def objective(space):
    clf = xgb.XGBClassifier(
        n_estimators=space['n_estimators'], max_depth=int(space['max_depth']), gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']), min_child_weight=int(space['min_child_weight']),
        colsample_bytree=int(space['colsample_bytree']),
        seed=space['seed']
    )

    clf.set_params(eval_metric="auc", early_stopping_rounds=10)

    evaluation = [(X_train, y_train), (X_test, y_test)]

    clf.fit(X_train, y_train,
            eval_set=evaluation,
            verbose=False)

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred > 0.5)
   # accuracy = roc_auc_score(y_test, pred)

    print("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}

trials = Trials()

best_hyperparams = fmin(fn=objective,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=200,
                        trials=trials)

print("The best hyperparameters are : ","\n")
print(best_hyperparams)

# Entraîner le modèle final avec les meilleurs hyperparamètres
best_clf = xgb.XGBClassifier(
    n_estimators=space['n_estimators'], max_depth=int(best_hyperparams['max_depth']), gamma=best_hyperparams['gamma'],
    reg_alpha=int(best_hyperparams['reg_alpha']), min_child_weight=int(best_hyperparams['min_child_weight']),
    colsample_bytree=int(best_hyperparams['colsample_bytree']),
    seed=space['seed']
)

best_clf.set_params(eval_metric="auc")

beval_set = [(X_test, y_test)]  # Définir l'ensemble de validation
best_clf.fit(X_train, y_train, eval_set=beval_set, verbose=False)

# Faire des prédictions sur l'ensemble de test
y_pred_proba = best_clf.predict_proba(X_test)[:, 1]  # Prédire les probabilités de la classe positive
y_pred = (y_pred_proba > 0.5).astype(int)  # Seuil de 0.5 pour la classification binaire

# Compter le nombre d'instances prédites pour chaque classe
predicted_classes = y_pred.tolist()
count_class_0 = predicted_classes.count(0)
count_class_1 = predicted_classes.count(1)

print("Nombre d'instances prédites pour la classe 0 :", count_class_0)
print("Nombre d'instances prédites pour la classe 1 :", count_class_1)
# Générer le rapport de classification
print(classification_report(y_test, y_pred))

# Compute correlation heatmap
def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', cmap="YlGnBu",
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70}
                )
    plt.show()

correlation_heatmap(X_train)

# Compute XGBoost built-in feature importance
sorted_idx = best_clf.feature_importances_.argsort()
plt.figure(figsize=(10, 6))
plt.barh(X.columns[sorted_idx], best_clf.feature_importances_[sorted_idx])
plt.xlabel("Xgboost Feature Importance")
plt.title("XGBoost Built-in Feature Importance")
plt.show()

# Compute permutation-based feature importance
perm_importance = permutation_importance(best_clf, X_test, y_test)
sorted_idx = perm_importance.importances_mean.argsort()
plt.figure(figsize=(10, 6))
plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Permutation-based Feature Importance")
plt.show()



# Compute feature importance using SHAP values
explainer = shap.TreeExplainer(best_clf)
shap_values = explainer.shap_values(X_test)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.xlabel("SHAP Feature Importance")
plt.title("Feature Importance using SHAP Values")
plt.show()