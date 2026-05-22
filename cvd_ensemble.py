"""
CVD Prediction — Stacking Ensemble
Datasets: cardio_train.csv (Kaggle, 70k) | heart.csv (Cleveland UCI, 303)

Just run:
    python cvd_ensemble.py

Pick a dataset from the menu, get results.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report
)


# ─────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────

class CardiacFeatureEngineer:
    """
    Derives composite risk indices from raw clinical features.
    These aren't just transformations — each one maps to a clinically
    validated concept (AIP, Framingham, metabolic syndrome, etc.)
    and forms a separate patent claim.
    """

    def fit(self, X):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if "triglycerides" in X.columns and "hdl" in X.columns:
            X["atherogenic_index"] = np.log10(
                (X["triglycerides"] + 1e-6) / (X["hdl"] + 1e-6)
            )

        if "total_cholesterol" in X.columns and "hdl" in X.columns:
            X["cardiac_risk_ratio"]  = X["total_cholesterol"] / (X["hdl"] + 1e-6)
            X["non_hdl_cholesterol"] = X["total_cholesterol"] - X["hdl"]

        if "systolic_bp" in X.columns and "diastolic_bp" in X.columns:
            X["pulse_pressure"]        = X["systolic_bp"] - X["diastolic_bp"]
            X["mean_arterial_pressure"]= X["diastolic_bp"] + X["pulse_pressure"] / 3

        if "bmi" in X.columns:
            X["bmi_category"] = pd.cut(
                X["bmi"], bins=[0, 18.5, 25, 30, 35, 100], labels=[0,1,2,3,4]
            ).astype(float)

        if "age" in X.columns and "systolic_bp" in X.columns:
            X["age_bp_interaction"] = X["age"] * (X["systolic_bp"] / 120)

        if all(c in X.columns for c in ["bmi","triglycerides","hdl","systolic_bp"]):
            X["metabolic_syndrome_score"] = (
                (X["bmi"] > 30).astype(int) +
                (X["triglycerides"] > 150).astype(int) +
                (X["hdl"] < 40).astype(int) +
                (X["systolic_bp"] > 130).astype(int)
            )

        if all(c in X.columns for c in ["age","total_cholesterol","hdl","systolic_bp"]):
            X["framingham_proxy"] = (
                0.04 * X["age"] +
                0.01 * X["total_cholesterol"] -
                0.02 * X["hdl"] +
                0.008 * X["systolic_bp"]
            )

        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ─────────────────────────────────────────────────────────────────
# STACKING ENSEMBLE
# ─────────────────────────────────────────────────────────────────

class CVDStackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Two-stage stacking ensemble.

    Stage 1 — five heterogeneous base learners trained with
    out-of-fold (OOF) predictions to prevent leakage:
        Logistic Regression, Random Forest, Gradient Boosting,
        SVM (RBF), MLP Neural Network

    Stage 2 — a meta-learner (Logistic Regression) trained on
    the OOF probability outputs from Stage 1.

    Final output includes a 4-tier risk label:
        Low (<25%) | Moderate (25-50%) | High (50-75%) | Critical (>75%)
    """

    def __init__(self, n_folds=5, calibrate=True):
        self.n_folds  = n_folds
        self.calibrate = calibrate
        self.engineer  = CardiacFeatureEngineer()
        self.imputer_  = SimpleImputer(strategy="median")
        self.scaler_   = StandardScaler()
        self.meta_     = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.base_models_ = {}
        self.classes_  = None

    def _base_definitions(self):
        return {
            "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=8,
                                       min_samples_split=5, random_state=42, n_jobs=-1),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200,
                                       learning_rate=0.05, max_depth=4, random_state=42),
            "SVM (RBF)":           SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
            "Neural Network":      MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                       activation="relu", max_iter=500,
                                       learning_rate_init=0.001, random_state=42,
                                       early_stopping=True, validation_fraction=0.1),
        }

    def _preprocess(self, X: pd.DataFrame, fit=False) -> np.ndarray:
        Xe = self.engineer.fit_transform(X) if fit else self.engineer.transform(X)
        if fit:
            Xi = self.imputer_.fit_transform(Xe)
            return self.scaler_.fit_transform(Xi)
        return self.scaler_.transform(self.imputer_.transform(Xe))

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_proc = self._preprocess(X, fit=True)
        y = np.array(y)
        base_defs = self._base_definitions()
        n = X_proc.shape[0]
        oof = np.zeros((n, len(base_defs)))
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        print(f"\n  Training {len(base_defs)} base models ({self.n_folds}-fold OOF)...")

        for fold, (tr, val) in enumerate(skf.split(X_proc, y)):
            for mi, (name, clf) in enumerate(base_defs.items()):
                m = CalibratedClassifierCV(clf, cv=3, method="isotonic") \
                    if self.calibrate and name != "Logistic Regression" else clf
                m.fit(X_proc[tr], y[tr])
                oof[val, mi] = m.predict_proba(X_proc[val])[:, 1]
            sys.stdout.write(f"\r  Fold {fold+1}/{self.n_folds} done  ")
            sys.stdout.flush()
        print()

        # Train final base models on full data
        for mi, (name, clf) in enumerate(base_defs.items()):
            m = CalibratedClassifierCV(clf, cv=3, method="isotonic") \
                if self.calibrate and name != "Logistic Regression" else clf
            m.fit(X_proc, y)
            self.base_models_[name] = m

        # Train meta-learner on OOF
        self.meta_.fit(oof, y)
        self.classes_ = np.unique(y)

        # Print OOF scores
        print(f"\n  {'Model':<25} {'OOF AUC':>9}")
        print("  " + "─" * 36)
        for mi, name in enumerate(base_defs):
            auc = roc_auc_score(y, oof[:, mi])
            print(f"  {name:<25} {auc:.4f}")
        ens_auc = roc_auc_score(y, self.meta_.predict_proba(oof)[:, 1])
        print("  " + "─" * 36)
        print(f"  {'Ensemble (meta-learner)':<25} {ens_auc:.4f}  ←")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_proc = self._preprocess(X, fit=False)
        base_p = np.column_stack([
            m.predict_proba(X_proc)[:, 1] for m in self.base_models_.values()
        ])
        return self.meta_.predict_proba(base_p)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba_per_model(self, X: pd.DataFrame) -> pd.DataFrame:
        X_proc = self._preprocess(X, fit=False)
        rows = {name: m.predict_proba(X_proc)[:, 1]
                for name, m in self.base_models_.items()}
        rows["Ensemble"] = self.predict_proba(X)[:, 1]
        return pd.DataFrame(rows)

    def risk_level(self, X: pd.DataFrame):
        p = self.predict_proba(X)[:, 1]
        def label(v):
            if v < 0.25: return "Low"
            if v < 0.50: return "Moderate"
            if v < 0.75: return "High"
            return "Critical"
        return [(label(v), v) for v in p]

    def feature_importances(self, cols):
        out = {}
        for name in ["Random Forest", "Gradient Boosting"]:
            m = self.base_models_.get(name)
            if m and hasattr(m, "feature_importances_"):
                out[name] = m.feature_importances_
        if not out:
            return pd.DataFrame()
        n = min(len(v) for v in out.values())
        df = pd.DataFrame(out, index=cols[:n])
        df["mean"] = df.mean(axis=1)
        return df.sort_values("mean", ascending=False)


# ─────────────────────────────────────────────────────────────────
# DATASET LOADERS
# ─────────────────────────────────────────────────────────────────

def load_cardio(path: str) -> pd.DataFrame:
    """
    cardio_train.csv — Kaggle cardiovascular dataset
    70,000 patients. Age stored in days. Separator is semicolon.
    """
    df = pd.read_csv(path, sep=";")
    df = df.drop(columns=["id"], errors="ignore")
    df["age"] = (df["age"] / 365.25).round(1)
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    df = df.drop(columns=["height", "weight"], errors="ignore")
    df = df.rename(columns={
        "ap_hi":       "systolic_bp",
        "ap_lo":       "diastolic_bp",
        "cholesterol": "cholesterol_level",
        "gluc":        "glucose_level",
        "smoke":       "smoking",
        "alco":        "alcohol",
        "active":      "physical_activity",
        "cardio":      "cvd",
    })
    # Map ordinal cholesterol/glucose to approximate mg/dL
    df["total_cholesterol"] = df["cholesterol_level"].map({1: 180, 2: 220, 3: 280})
    df["glucose"]           = df["glucose_level"].map({1: 90, 2: 130, 3: 200})
    df = df.drop(columns=["cholesterol_level", "glucose_level"])

    # Remove obvious outliers (BP values that are physiologically impossible)
    df = df[(df["systolic_bp"] > 50) & (df["systolic_bp"] < 250)]
    df = df[(df["diastolic_bp"] > 30) & (df["diastolic_bp"] < 200)]

    return df


def load_heart(path: str) -> pd.DataFrame:
    """
    heart.csv — Cleveland UCI Heart Disease
    303 patients. All standard column names with header row.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    df = df.rename(columns={
        "sex":      "gender",
        "trestbps": "systolic_bp",
        "chol":     "total_cholesterol",
        "fbs":      "diabetes",
        "thalach":  "max_heart_rate",
        "exang":    "exercise_angina",
        "oldpeak":  "st_depression",
        "target":   "cvd",
    })
    # Convert target: >0 means disease present
    df["cvd"] = (df["cvd"] > 0).astype(int)
    df = df.dropna()
    return df


# ─────────────────────────────────────────────────────────────────
# EVALUATION PRINTER
# ─────────────────────────────────────────────────────────────────

def print_metrics(model, X_test, y_test, dataset_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*56}")
    print(f"  RESULTS — {dataset_name}")
    print(f"{'='*56}")
    print(f"  Accuracy          {acc:.4f}   ({acc*100:.1f}%)")
    print(f"  Precision         {prec:.4f}")
    print(f"  Recall            {rec:.4f}")
    print(f"  F1 Score          {f1:.4f}")
    print(f"  ROC-AUC           {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted")
    print(f"               No CVD    CVD")
    print(f"  Actual No CVD  {tn:5d}   {fp:5d}   (specificity {tn/(tn+fp):.3f})")
    print(f"  Actual CVD     {fn:5d}   {tp:5d}   (sensitivity {tp/(tp+fn):.3f})")
    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["No CVD", "CVD"])
    for line in report.splitlines():
        print("    " + line)

    # Per-model breakdown on a sample
    print(f"  Per-model probabilities (first 6 test patients):")
    pm = model.predict_proba_per_model(X_test.head(6))
    print(pm.round(3).to_string(index=False))
    print()

    # Risk stratification
    print(f"  Risk stratification (first 12 test patients):")
    print(f"  {'#':<5} {'Risk Level':<12} {'Prob':>7}  {'Bar'}")
    print("  " + "─"*48)
    icons = {"Low":"·","Moderate":"▪","High":"▸","Critical":"▶"}
    risks = model.risk_level(X_test.head(12))
    for i, (level, prob) in enumerate(risks):
        bar = "█" * int(prob * 24)
        ic  = icons.get(level, " ")
        print(f"  {i+1:<5} {ic} {level:<10}  {prob*100:5.1f}%  {bar}")

    print(f"\n{'='*56}\n")


def print_feature_importance(model, X_train):
    eng   = CardiacFeatureEngineer()
    X_eng = eng.transform(X_train)
    fi    = model.feature_importances(list(X_eng.columns))
    if fi.empty:
        return
    print("  Top 10 feature importances (tree models):")
    print(f"  {'Feature':<30} {'RF':>7}  {'GB':>7}  {'Mean':>7}")
    print("  " + "─"*52)
    cols = [c for c in ["Random Forest","Gradient Boosting"] if c in fi.columns]
    for feat, row in fi.head(10).iterrows():
        rf_v = f"{row['Random Forest']:.4f}" if "Random Forest" in fi.columns else "  —  "
        gb_v = f"{row['Gradient Boosting']:.4f}" if "Gradient Boosting" in fi.columns else "  —  "
        print(f"  {str(feat):<30} {rf_v}  {gb_v}  {row['mean']:.4f}")
    print()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

DATASETS = {
    "1": ("cardio_train.csv  (Kaggle, 70k patients)", "cardio"),
    "2": ("heart.csv         (Cleveland UCI, 303 patients)", "heart"),
    "3": ("Both datasets (train & evaluate on each)", "both"),
}


def pick_dataset():
    print("""
╔══════════════════════════════════════════╗
║   CVD Prediction — Stacking Ensemble    ║
╚══════════════════════════════════════════╝

  Which dataset do you want to run on?
""")
    for key, (label, _) in DATASETS.items():
        print(f"    [{key}]  {label}")
    print()

    while True:
        choice = input("  Enter 1, 2 or 3: ").strip()
        if choice in DATASETS:
            return DATASETS[choice][1]
        print("  Not a valid choice, try again.")


def resolve_path(filename):
    """Look for dataset in common locations."""
    candidates = [
        Path(filename),
        Path(".") / filename,
        Path("/mnt/user-data/uploads") / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def run_dataset(name, loader_fn, csv_name, label):
    path = resolve_path(csv_name)
    if path is None:
        print(f"\n  File not found: {csv_name}")
        print(f"  Put it in the same folder as this script.\n")
        return

    print(f"\n  Loading {label}...")
    df = loader_fn(path)

    # Coerce all to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["cvd"])
    df["cvd"] = df["cvd"].astype(int)

    print(f"  Rows: {len(df):,}  |  Features: {df.shape[1]-1}")
    print(f"  CVD positive: {df['cvd'].sum():,} ({df['cvd'].mean()*100:.1f}%)")

    X = df.drop(columns=["cvd"])
    y = df["cvd"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    model = CVDStackingEnsemble(n_folds=5, calibrate=True)
    model.fit(X_train, y_train)

    print_metrics(model, X_test, y_test, label)
    print_feature_importance(model, X_train)


def main():
    choice = pick_dataset()

    if choice in ("cardio", "both"):
        run_dataset(
            name="cardio",
            loader_fn=load_cardio,
            csv_name="cardio_train.csv",
            label="Kaggle Cardiovascular (70k)"
        )

    if choice in ("heart", "both"):
        run_dataset(
            name="heart",
            loader_fn=load_heart,
            csv_name="heart.csv",
            label="Cleveland UCI Heart Disease (303)"
        )


if __name__ == "__main__":
    main()