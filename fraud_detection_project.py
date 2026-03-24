# ==============================
# CREDIT CARD FRAUD DETECTION
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==============================
# LOAD DATASET (AUTO DOWNLOAD)
# ==============================

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
data = pd.read_csv(url)

print("\n Dataset Loaded Successfully!\n")

# ==============================
# BASIC INFO
# ==============================

print("Shape:", data.shape)
print("\nColumns:\n", data.columns)
print("\nMissing Values:\n", data.isnull().sum())

# ==============================
# CLASS DISTRIBUTION
# ==============================

print("\nFraud vs Normal:\n", data['Class'].value_counts())

sns.countplot(x='Class', data=data)
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

# ==============================
# FEATURE SCALING
# ==============================

scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])

# Drop Time column
data = data.drop(['Time'], axis=1)

# ==============================
# SPLIT DATA
# ==============================

X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# MODEL 1: LOGISTIC REGRESSION
# ==============================

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\n Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# ==============================
# MODEL 2: RANDOM FOREST (FAST VERSION)
# ==============================

rf = RandomForestClassifier(n_estimators=10, max_depth=10, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# FINAL RESULT
# ==============================

print("\n Project Completed Successfully!")