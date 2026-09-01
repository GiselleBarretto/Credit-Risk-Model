import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Load & prepare ──────────────────────────────────────────────
df = pd.read_csv("data/loan_data.csv")
df = pd.get_dummies(df, columns=["loan_purpose"], drop_first=True)

X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Default rate: {y.mean():.2%}\n")

# ── Model ───────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    random_state=42,
    eval_metric="auc",
    early_stopping_rounds=20,
    verbosity=0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# ── Evaluation ──────────────────────────────────────────────────
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="YlOrBr")
ax.set_title("Credit Risk Model - Confusion Matrix")
plt.tight_layout()
plt.savefig("data/confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved to data/confusion_matrix.png")

# ── SHAP Explainability ─────────────────────────────────────────
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:500])

plt.figure()
shap.summary_plot(shap_values, X_test[:500], show=False, plot_type="bar")
plt.tight_layout()
plt.savefig("data/shap_summary.png", dpi=150, bbox_inches="tight")
print("SHAP summary saved to data/shap_summary.png")

# ── Save model ──────────────────────────────────────────────────
joblib.dump(model, "data/credit_risk_model.pkl")
joblib.dump(list(X.columns), "data/feature_names.pkl")
print("\nModel saved to data/credit_risk_model.pkl")
