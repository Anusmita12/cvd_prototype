# IMPORTS 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from imblearn.over_sampling import SMOTE
import xgboost as xgb

# LOAD DATA 
dataset = pd.read_csv("heart.csv")

# DATA QUALITY 
# Remove duplicates
dataset = dataset.drop_duplicates()

# Check missing values
print(dataset.isnull().sum())

# FEATURES & TARGET 
X = dataset.drop("target", axis=1)
y = dataset["target"]

# SCALING 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# CLASS IMBALANCE (FIXED)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# TRAIN TEST SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# MODELS 
lr = LogisticRegression(max_iter=2000)
sv = svm.SVC(kernel='linear', probability=True)
rf = RandomForestClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

# ENSEMBLE 
ensemble = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('svm', sv),
        ('rf', rf),
        ('xgb', xgb_model)
    ],
    voting='soft'
)

# TRAIN
ensemble.fit(X_train, y_train)

# PREDICT 
y_pred = ensemble.predict(X_test)

# EVALUATION 
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# CROSS VALIDATION (GENERALIZATION) 
cv_scores = cross_val_score(ensemble, X_resampled, y_resampled, cv=5)
print("\nCross Validation Accuracy:", cv_scores.mean())

# FEATURE IMPORTANCE 
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10,5))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.show()
# Risk Categorization
y_prob = ensemble.predict_proba(X_test)[:, 1]
def risk_category(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

# Apply to predictions
risk_levels = [risk_category(p) for p in y_prob]

# Show few results
for i in range(5):
    print(f"Patient {i+1}: Probability={round(y_prob[i],2)}, Risk={risk_levels[i]}")