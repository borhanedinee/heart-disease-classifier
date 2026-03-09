import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── 1. REPRODUCIBILITY ───────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── 2. LOAD DATA ─────────────────────────────────────────────────
df = pd.read_csv("data/heart.csv")

# ── 3. DATA CLEANING ─────────────────────────────────────────────
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)
df = df.astype(float)

# Binary classification: 0 = no disease, 1 = disease
df["target"] = (df["target"] > 0).astype(int)

print(f"Dataset shape after cleaning: {df.shape}")
print(f"Class distribution:\n{df['target'].value_counts()}")

# ── 4. SPLIT ─────────────────────────────────────────────────────
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

print(f"\nSplit → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── 5. SCALING ───────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ── 6. TRAIN MODEL ───────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=SEED
)
model.fit(X_train, y_train)

# ── 7. EVALUATE ON VALIDATION SET ────────────────────────────────
val_preds = model.predict(X_val)
val_probs = model.predict_proba(X_val)[:, 1]

f1    = f1_score(y_val, val_preds)
auroc = roc_auc_score(y_val, val_probs)

print(f"\n── Validation Results ──────────────────────")
print(f"F1 Score : {f1:.4f}")
print(f"AUROC    : {auroc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_val, val_preds)}")

# ── 8. CONFUSION MATRIX ──────────────────────────────────────────
cm = confusion_matrix(y_val, val_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Confusion Matrix - Validation Set")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix saved → confusion_matrix.png")

# ── 9. FEATURE IMPORTANCE ────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=df.drop("target", axis=1).columns)
importances.sort_values().plot(kind="barh", figsize=(8, 6))
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Feature importance saved → feature_importance.png")

print("\n✅ Training complete. Checkpoint: RandomForest_seed42_depth6.pkl")


import joblib
joblib.dump(model, "model.pkl")
print("Model saved → model.pkl")