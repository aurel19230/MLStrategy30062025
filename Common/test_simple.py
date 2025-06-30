import xgboost as xgb
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cupy as cp

# Charger le dataset
data = fetch_covtype()
X, y = data.data, data.target

# Ajuster les étiquettes pour qu'elles soient dans la plage [0, 6]
y = y - 1

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Définir les hyperparamètres manuellement
params = {
    'objective': 'multi:softprob',
    'num_class': 7,
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 200,
    'tree_method': 'hist',
    'random_state': 42,
    'device': 'gpu',
    'use_label_encoder': False
}

# Créer et entraîner le modèle XGBoost
model = xgb.XGBClassifier(**params)
X_train = cp.array(X_train)
model.fit(X_train, y_train,verbose=True)

# Faire des prédictions avec le modèle entraîné
X_test = cp.array(X_test)

preds = model.predict(X_test)

# Calculer les métriques
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average=None, zero_division=0)
recall = recall_score(y_test, preds, average=None, zero_division=0)
f1 = f1_score(y_test, preds, average=None, zero_division=0)

# Afficher les métriques pour chaque classe
print(f'Test Accuracy: {accuracy:.4f}')
for cls in range(7):
    print(f'Test Precision for class {cls}: {precision[cls]:.4f}')
    print(f'Test Recall for class {cls}: {recall[cls]:.4f}')
    print(f'Test F1 Score for class {cls}: {f1[cls]:.4f}')
