import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Chemin d'accès au fichier
file_path = "C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/Sierra chart/xTickReversal/data_simuNew/example/candlestick_data.csv"

# Vérifier si le fichier existe
if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    print(data.head())
else:
    print(f"Le fichier {file_path} n'existe pas.")

# Remove class 1 (No Trade) and relabel classes 0 and 2
data_filtered = data[data['trade_status'] != 1]
data_filtered['trade_status'] = data_filtered['trade_status'].apply(lambda x: 1 if x == 2 else 0)

# Check class distribution
print(data_filtered['trade_status'].value_counts())

# Separate features and target
X = data_filtered.drop('trade_status', axis=1)
y = data_filtered['trade_status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a binary classifier (Logistic Regression in this example)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test_scaled)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
