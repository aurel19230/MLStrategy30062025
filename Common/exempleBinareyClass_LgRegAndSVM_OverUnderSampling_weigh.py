import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Chemin d'accès au fichier
file_path = "C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/Sierra chart/xTickReversal/data_simuNew/example/loan_data.csv"

# Vérifier si le fichier existe
if os.path.exists(file_path):
    loan_data = pd.read_csv(file_path)
    print(loan_data.head())
else:
    print(f"Le fichier {file_path} n'existe pas.")


# Visualize the proportion of borrowers
def show_loan_distrib(data):
    if isinstance(data, pd.DataFrame):
        count = data["not.fully.paid"].value_counts()
    else:
        count = data.value_counts()
    count.plot(kind='pie', explode=[0, 0.1], figsize=(6, 6), autopct='%1.1f%%', shadow=True)
    plt.ylabel("Loan: Fully Paid Vs. Not Fully Paid")
    plt.legend(["Fully Paid", "Not Fully Paid"])
    plt.show()


show_loan_distrib(loan_data)

# Check for null values.
print(loan_data.isnull().sum())

# Check column types
print(loan_data.dtypes)

encoded_loan_data = pd.get_dummies(loan_data, prefix="purpose", drop_first=True)
print(encoded_loan_data.dtypes)

X = encoded_loan_data.drop('not.fully.paid', axis=1)
y = encoded_loan_data['not.fully.paid']

# 1. Undersampling
X_train_cp = X.copy()
X_train_cp['not.fully.paid'] = y

y_0 = X_train_cp[X_train_cp['not.fully.paid'] == 0]
y_1 = X_train_cp[X_train_cp['not.fully.paid'] == 1]

y_0_undersample = y_0.sample(y_1.shape[0])
loan_data_undersample = pd.concat([y_0_undersample, y_1], axis=0)

show_loan_distrib(loan_data_undersample)

# 2. SMOTE Oversampling
smote = SMOTE(sampling_strategy='minority')
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X, y)

show_loan_distrib(y_train_SMOTE)

# Choose between original data, oversampled data, or undersampled data
data_choice = input("Choose the type of data to use (o: original, v: oversampled, u: undersampled): ")

if data_choice.lower() == 'o':
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=2022)
    print("Using original data.")

    # Ask if user wants to use class weights with original data
    use_class_weights = input("Do you want to use class weights? (y/n): ")
    class_weight = 'balanced' if use_class_weights.lower() == 'y' else None

elif data_choice.lower() == 'v':
    X_train, X_test, y_train, y_test = train_test_split(X_train_SMOTE, y_train_SMOTE, test_size=0.15,
                                                        stratify=y_train_SMOTE, random_state=2022)
    print("Using oversampled data.")
    class_weight = None  # No class weights needed for oversampled data

else:
    X_train, X_test, y_train, y_test = train_test_split(loan_data_undersample.drop('not.fully.paid', axis=1),
                                                        loan_data_undersample['not.fully.paid'],
                                                        test_size=0.15,
                                                        stratify=loan_data_undersample['not.fully.paid'],
                                                        random_state=2022)
    print("Using undersampled data.")
    class_weight = None  # No class weights needed for undersampled data

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Logistic Regression with Conditional Class Weights
logistic_classifier = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=0, n_jobs=-1)
logistic_classifier.fit(X_train_scaled, y_train)
y_pred = logistic_classifier.predict(X_test_scaled)

print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred))

# 2. SVM with Conditional Class Weights
svc_classifier = SVC(kernel='linear', class_weight=class_weight, random_state=0, n_jobs=-1)
svc_classifier.fit(X_train_scaled, y_train)
y_pred = svc_classifier.predict(X_test_scaled)

print("SVM Classification Report:")
print(classification_report(y_test, y_pred))
