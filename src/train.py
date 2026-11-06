"""Clean data -> EDA -> feature engineering -> train & compare 3 models."""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features import CATEGORICAL, FEATURES, NUMERIC, add_features, clean

warnings.filterwarnings("ignore")
FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# ---------- 1. load + clean ----------
raw = pd.read_csv("data/raw/customers.csv")
df = clean(raw)
print(f"Raw rows: {len(raw)} -> clean rows: {len(df)} | churn rate: {df['churn'].mean():.1%}")

# ---------- 2. EDA ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df.groupby("contract")["churn"].mean().mul(100).plot.bar(ax=axes[0], color="#4C78A8")
axes[0].set_title("Churn rate by contract (%)")
df.groupby("internet_service")["churn"].mean().mul(100).plot.bar(ax=axes[1], color="#F58518")
axes[1].set_title("Churn rate by internet service (%)")
df.boxplot(column="tenure", by="churn", ax=axes[2])
axes[2].set_title("Tenure by churn (0=stayed, 1=churned)")
plt.suptitle("")
plt.tight_layout()
plt.savefig(FIG / "eda.png", dpi=120)
plt.close()

# ---------- 3. feature engineering ----------
df = add_features(df)
X, y = df[FEATURES], df["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pre = ColumnTransformer(
    [("num", StandardScaler(), NUMERIC),
     ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)]
)

# ---------- 4. models ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                            class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=5,
                                            class_weight="balanced", random_state=42, n_jobs=-1),
}

rows, fitted, cms = [], {}, {}
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("model", clf)]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    })
    fitted[name], cms[name] = pipe, confusion_matrix(y_test, pred)

metrics = pd.DataFrame(rows).round(3).sort_values("f1", ascending=False)
metrics.to_csv("reports/metrics.csv", index=False)
print("\n", metrics.to_string(index=False))

# confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, cm) in zip(axes, cms.items()):
    ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig(FIG / "confusion_matrices.png", dpi=120)
plt.close()

# ---------- 5. save best model + predictions ----------
best_name = metrics.iloc[0]["model"]
best = fitted[best_name]
joblib.dump(best, "models/best_model.joblib")
json.dump({"best_model": best_name}, open("models/best_model.json", "w"))

out = df.copy()
out["churn_probability"] = best.predict_proba(X)[:, 1].round(4)
out["split"] = "train"
out.loc[X_test.index, "split"] = "test"
out.to_csv("data/processed/customers_scored.csv", index=False)
out.drop(columns=["tenure_group", "avg_monthly_spend", "num_services",
                  "churn_probability", "split"]).assign(
    churn=lambda d: d["churn"].map({1: "Yes", 0: "No"})
).to_csv("data/processed/customers_clean.csv", index=False)

# feature importance (Random Forest) for the dashboard
rf = fitted["Random Forest"]
names = rf.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
imp.head(15).rename_axis("feature").reset_index(name="importance").to_csv(
    "reports/feature_importance.csv", index=False)
print(f"\nBest model: {best_name}  (saved to models/best_model.joblib)")
