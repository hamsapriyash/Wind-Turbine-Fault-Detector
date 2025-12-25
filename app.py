# train_model_with_smote.py
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import shap

RANDOM_STATE = 42
TRAIN_PATH = "Train.csv"   # put your Train.csv here
MODEL_PATH = "final_model.joblib"
SCALER_PATH = "scaler.joblib"
EXPLAINER_PATH = "shap_explainer.joblib"

# 1. Load data
df = pd.read_csv(TRAIN_PATH)
print("Shape:", df.shape)

# 2. Binarize target (0 normal, 1 fault)
if "Target" not in df.columns:
    raise ValueError("Train.csv must contain 'Target'")
df["Target_bin"] = (df["Target"] != 0).astype(int)
print("Original class counts:", df["Target_bin"].value_counts())

# 3. Features = all V1..V40
feature_cols = [c for c in df.columns if c.startswith("V")]
if len(feature_cols) != 40:
    print("Warning: expected 40 V-features, found:", len(feature_cols))

X = df[feature_cols].copy()
y = df["Target_bin"].copy()

# 4. Handle missing values (median)
X = X.fillna(X.median())

# 5. Train/validation split (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)
print("Train distribution before SMOTE:", Counter(y_train))
print("Val distribution:", Counter(y_val))

# 6. Apply SMOTE to training data
sm = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("After SMOTE:", Counter(y_train_res))

# 7. Optional scaler (not strictly needed for RF, but OK to keep)
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_val_scaled = scaler.transform(X_val)

# 8. Train RandomForest on ALL 40 features
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
    n_jobs=-1,
)
rf.fit(X_train_res_scaled, y_train_res)

# 9. Evaluate
y_pred = rf.predict(X_val_scaled)
if hasattr(rf, "predict_proba"):
    y_prob = rf.predict_proba(X_val_scaled)[:, 1]
else:
    y_prob = None

print("\nValidation classification report:")
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
print("Confusion matrix:\n", cm)

if y_prob is not None:
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    pr_auc = auc(recall, precision)
    print("PR-AUC:", pr_auc)

# 10. Save model + scaler
joblib.dump(rf, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print("Saved model to", MODEL_PATH)
print("Saved scaler to", SCALER_PATH)

# # 11. Build SHAP explainer on a sample of training data
# print("Fitting SHAP explainer on a subset...")
# sample_idx = np.random.choice(len(X_train_res_scaled), size=min(2000, len(X_train_res_scaled)), replace=False)
# X_sample = X_train_res_scaled[sample_idx]

# explainer = shap.TreeExplainer(rf)
# _ = explainer.shap_values(X_sample)  # warm up

# joblib.dump(explainer, EXPLAINER_PATH)
# print("Saved SHAP explainer to", EXPLAINER_PATH)
# print("Done.")
