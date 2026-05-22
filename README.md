# Cardiovascular Disease Prediction System

This project is a machine learning-based system designed to predict the likelihood of cardiovascular disease using patient health data. The model combines multiple algorithms through ensemble learning to improve prediction accuracy and reliability.

## Features
- Data preprocessing and cleaning
- Feature scaling using StandardScaler
- Class imbalance handling with SMOTE
- Ensemble model using:
  - Logistic Regression
  - SVM
  - Random Forest
  - XGBoost
- Performance evaluation using accuracy score, confusion matrix, and classification report
- Feature importance visualization
- Risk categorization (Low, Medium, High)

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## How to Run

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

Run the Python file:

```bash
python prototype.py
```

---

## Output

The project generates:

- Prediction accuracy
- Confusion matrix
- Classification report
- Cross-validation accuracy
- Feature importance visualization
- Patient risk category prediction:
  - Low Risk
  - Medium Risk
  - High Risk
