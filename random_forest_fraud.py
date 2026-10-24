# Fraud detection baseline model for FraudDetection_Dataset.csv
# Run from the folder containing this script and the CSV file.

import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

DATA_FILE = "FraudDetection_Dataset.csv"
TARGET = "fraud_bool"

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found. Columns: {list(df.columns)}")

# The dataset has month values 0-7. Keep the chronological split used by the paper:
# months 0-5 for training, months 6-7 for testing.
if "month" in df.columns and set(df["month"].dropna().unique()).issubset(set(range(8))):
    train_df = df[df["month"] <= 5].copy()
    test_df = df[df["month"] >= 6].copy()
    print(f"Chronological split: train months 0-5 ({len(train_df):,} rows), test months 6-7 ({len(test_df):,} rows)")
else:
    # Fallback for a differently formatted dataset.
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])
    print(f"Random stratified split: train ({len(train_df):,}), test ({len(test_df):,})")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET].astype(int)
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET].astype(int)

# Keep month as a feature: the split is chronological, but month can still signal time effects.
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False)),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, num_cols),
    ("categorical", categorical_pipe, cat_cols),
])

MODEL_FILE = "random_forest_fraud_model.joblib"

# Balanced subsampling helps the model learn from the rare fraud class.
model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )),
])

print("Training Random Forest model... This can take several minutes.")
start = time.time()
model.fit(X_train, y_train)
print(f"Training completed in {(time.time()-start)/60:.2f} minutes")

print("\nEvaluating on unseen test months...")
y_pred = model.predict(X_test)
if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X_test)[:, 1]
else:
    y_score = y_pred

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) else 0.0
print("\n===== TEST RESULTS =====")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
try:
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_score):.4f}")
except ValueError:
    print("ROC-AUC  : unavailable (test set contains one class)")
print(f"False Positive Rate (FPR): {fpr:.4f} ({fpr*100:.2f}%)")
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(cm)
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))
joblib.dump(model, MODEL_FILE)
print(f"Saved trained model to: {MODEL_FILE}")
