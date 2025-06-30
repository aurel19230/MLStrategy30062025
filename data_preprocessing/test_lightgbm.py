import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import cupy as cp

# Charger les données
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_cp = cp.asarray(X_train)
X_test_cp = cp.asarray(X_test)
y_train_cp = cp.asarray(y_train)
y_test_cp = cp.asarray(y_test)
# Créer les Datasets
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

# Fonction objective personnalisée
def custom_objective(preds, train_data):
    y_true = train_data.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - y_true
    hess = preds * (1.0 - preds)
    return grad, hess

# Custom metric
def custom_metric(preds, train_data):
    y_true = train_data.get_label()
    preds_binary = (preds > 0.5).astype(int)
    accuracy = np.mean(y_true == preds_binary)
    return 'custom_accuracy', accuracy, True

params = {
    'objective': custom_objective,
    'metric': 'None',       # Désactiver toutes les métriques par défaut
    'verbose': 1
}

# Dictionnaire pour stocker les résultats d'évaluation
evals_result = {}

# Utiliser un callback pour enregistrer les résultats d'évaluation
model = lgb.train(
    params=params,
    train_set=train_data,
    valid_sets=[train_data, valid_data],
    valid_names=['training', 'validation'],
    num_boost_round=50,
    feval=custom_metric,
    callbacks=[lgb.record_evaluation(evals_result)]  # Enregistre les métriques à chaque itération
)

# Afficher les résultats directement pour la custom metric
print("Métriques sur l'ensemble d'entraînement :")
print("custom_accuracy:", evals_result['training']['custom_accuracy'])

print("\nMétriques sur l'ensemble de validation :")
print("custom_accuracy:", evals_result['validation']['custom_accuracy'])


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Prédiction sur le jeu de test (probabilités)
y_probs = model.predict(X_test)

# Application d'un seuil pour obtenir des prédictions binaires
threshold = 0.5
y_pred = (y_probs > threshold).astype(int)

# Calcul de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

# Calcul d'autres métriques
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)  # zero_division=0 pour éviter les erreurs division par 0
recall = recall_score(y_test, y_pred, zero_division=0)

print("\nMatrice de confusion:")
print(cm)
print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
print(f"Accuracy: {accuracy:.4f}, Précision: {precision:.4f}, Rappel: {recall:.4f}")