from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

df = pd.read_csv("data/heart.csv")
X, y = df[NUMERIC + CATEGORICAL], df["target"]

# Hold out 20% that the model never sees during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Numbers: fill gaps with the median, then scale
# Categories: fill gaps with the most common value, then one-hot encode
preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), NUMERIC),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), CATEGORICAL),
])

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
}

# Compare models with 5-fold cross-validation on the training set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_name, best_auc = None, 0.0
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    s = cross_validate(pipe, X_train, y_train, cv=cv,
                       scoring=["roc_auc", "recall", "accuracy"])
    auc = s["test_roc_auc"].mean()
    print(f"{name}: AUC={auc:.3f}  recall={s['test_recall'].mean():.3f}  "
          f"accuracy={s['test_accuracy'].mean():.3f}")
    if auc > best_auc:
        best_name, best_auc = name, auc

# Retrain the winner on all training data, then test once
best = Pipeline([("prep", preprocess), ("model", models[best_name])])
best.fit(X_train, y_train)
pred = best.predict(X_test)
proba = best.predict_proba(X_test)[:, 1]

print(f"\nBest model: {best_name}\n")
print(classification_report(y_test, pred, target_names=["no disease", "disease"]))
print("Test AUC:", round(roc_auc_score(y_test, proba), 3))

Path("models").mkdir(exist_ok=True)
joblib.dump(best, "models/heart_model.joblib")
print("Saved models/heart_model.joblib")