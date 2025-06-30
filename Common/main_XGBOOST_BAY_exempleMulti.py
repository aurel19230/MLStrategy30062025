
import xgboost as xgb
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cupy as cp
from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval
from sklearn.utils.class_weight import compute_class_weight


xgboost1 = XGBClassifier()

print(xgboost1.get_params())


# Charger le dataset
data = datasets.fetch_covtype()

# Mettre les données au format DataFrame de pandas
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target
y = df['target']

print(y.value_counts(normalize=True))

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df[df.columns.difference(['target'])],
                                                    df['target'],
                                                    test_size=0.2,
                                                    random_state=42)

# Ajuster les étiquettes pour qu'elles soient dans la plage [0, 6]
y_train = y_train - 1
y_test = y_test - 1

# Normaliser les données
scaler = StandardScaler()

# Standardiser l'ensemble d'entraînement
X_train_transformed = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)

# Standardiser l'ensemble de test
X_test_transformed = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

print("\n------------------Optimisation Bayésienne pour XGBoost-------------------------------\n")

space = {
    'learning_rate': hp.choice('learning_rate', [0.0001, 0.001, 0.01, 0.1, 1]),
    'max_depth': hp.choice('max_depth', range(3, 21, 3)),
    'gamma': hp.choice('gamma', [i / 10.0 for i in range(0, 5)]),
    'colsample_bytree': hp.choice('colsample_bytree', [i / 10.0 for i in range(3, 10)]),
    'reg_alpha': hp.choice('reg_alpha', [1e-5, 1e-2, 0.1, 1, 10, 100]),
    'reg_lambda': hp.choice('reg_lambda', [1e-5, 1e-2, 0.1, 1, 10, 100])
}

# Configurer la validation croisée k-fold
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

# Calculer les poids de classe
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

def objective(params):
    params['tree_method'] = 'hist'
    params['device'] = 'cuda'
    params['verbosity'] = 2

    xgboost = xgb.XGBClassifier(seed=0, **params)

    X_train_cp = cp.array(X_train_transformed)

    cv_scores = cross_val_score(estimator=xgboost,
                                X=X_train_cp,
                                y=y_train,
                                cv=kfold,
                                scoring='recall_weighted',
                                n_jobs=1)

    cv_score = cv_scores.mean()

    xgboost.fit(X_train_cp, y_train)
    train_score = xgboost.score(X_train_cp, y_train)

    loss = - cv_score

    return {'loss': loss, 'params': params, 'status': STATUS_OK, 'train_score': train_score, 'cv_score': cv_score}


# Optimiser
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=48,
            trials=trials,
            verbose=True)


print("\n------------------Afficher les scores pour chaque essai-------------------------------")


# Afficher les scores pour chaque essai
for trial in trials.trials:
    train_score = trial['result']['train_score']
    cv_score = trial['result']['cv_score']
    print(f"Score d'entraînement: {train_score:.4f}, Score de validation croisée: {cv_score:.4f}, Différence: {(train_score - cv_score):.4f}")

print("\n------------------Afficher les meilleurs hyperparamètres et leur différence de score------------------------------")
# Afficher les meilleurs hyperparamètres et leur différence de score
best_params = space_eval(space, best)
best_trial_index = np.argmin([t['result']['loss'] for t in trials.trials])
best_trial = trials.trials[best_trial_index]
best_train_score = best_trial['result']['train_score']
best_cv_score = best_trial['result']['cv_score']
print(f"Meilleurs hyperparamètres trouvés: {best_params}")
print(f"Score d'entraînement: {best_train_score:.4f}, Score de validation croisée: {best_cv_score:.4f}, Différence: {(best_train_score - best_cv_score):.4f}")

print("\n------------------# Entraîner le modèle final avec les meilleurs hyperparamètres------------------------------")

# Entraîner le modèle final avec les meilleurs hyperparamètres
best_params['objective'] = 'multi:softprob'
best_params['num_class'] = 7
best_params['eval_metric'] = 'mlogloss'
best_params['n_estimators'] = 200
best_params['tree_method'] = 'hist'
best_params['device'] = 'cuda'
best_params['random_state'] = 42
best_params['use_label_encoder'] = False

model = xgb.XGBClassifier(**best_params)

X_train_cp = cp.array(X_train_transformed)
model.fit(X_train_cp, y_train, verbose=True)

# Faire des prédictions sur l'ensemble de test
X_test_cp = cp.array(X_test_transformed)
preds = model.predict(X_test_cp)

# Calculer les métriques
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average=None, zero_division=0)
recall = recall_score(y_test, preds, average=None, zero_division=0)
f1 = f1_score(y_test, preds, average=None, zero_division=0)

# Afficher les métriques pour chaque classe
print(f'Accuracy sur le test: {accuracy:.4f}')
for cls in range(7):
    print(f'Precision sur le test pour la classe {cls}: {precision[cls]:.4f}')
    print(f'Recall sur le test pour la classe {cls}: {recall[cls]:.4f}')
    print(f'F1 Score sur le test pour la classe {cls}: {f1[cls]:.4f}')

'''''
import xgboost as xgb
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    # Load the dataset
data = datasets.fetch_covtype()

    # Put the data in pandas dataframe format
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target']=data.target
y=df['target']

print(y.value_counts(normalize=True))

    # Train test split
X_train, X_test, y_train, y_test = train_test_split(df[df.columns.difference(['target'])],
                                                        df['target'],
                                                        test_size=0.2,
                                                        random_state=42)

    # Adjust labels to be in the range [0, 6]
y_train = y_train - 1
y_test = y_test - 1

    # Normalize the data
scaler = StandardScaler()

    # Standardize the training dataset
X_train_transformed = pd.DataFrame(scaler.fit_transform(X_train),index=X_train.index, columns=X_train.columns)

    # Standardized the testing dataset
X_test_transformed = pd.DataFrame(scaler.transform(X_test),index=X_test.index, columns=X_test.columns)

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
    params['tree_method'] = 'hist'

    xgboost = XGBClassifier(seed=0, **params)
    score = cross_val_score(estimator=xgboost,
                                X=X_train_transformed,
                                y=y_train,
                                cv=kfold,
                                scoring='recall_weighted',
                                n_jobs=1).mean()

        # Loss is negative score
    loss = - score

        # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

    # Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = 48, trials = Trials())

    # Print the best parameters
print("Best hyperparameters found:", space_eval(space, best))

    # Train the final model with the best hyperparameters
best_params = space_eval(space, best)
best_params['objective'] = 'multi:softprob'
best_params['num_class'] = 7
best_params['eval_metric'] = 'mlogloss'
best_params['n_estimators'] = 200
best_params['tree_method'] = 'hist'
best_params['random_state'] = 42
best_params['use_label_encoder'] = False

model = xgb.XGBClassifier(**best_params)
model.fit(X_train_transformed, y_train, verbose=True)

    # Make predictions on the test set
preds = model.predict(X_test_transformed)

    # Calculate metrics
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average=None, zero_division=0)
recall = recall_score(y_test, preds, average=None, zero_division=0)
f1 = f1_score(y_test, preds, average=None, zero_division=0)

    # Display metrics for each class
print(f'Test Accuracy: {accuracy:.4f}')
for cls in range(7):
    print(f'Test Precision for class {cls}: {precision[cls]:.4f}')
    print(f'Test Recall for class {cls}: {recall[cls]:.4f}')
    print(f'Test F1 Score for class {cls}: {f1[cls]:.4f}')
'''''