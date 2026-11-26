from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.features import make_features
from src.data.ingest import load_data

Path("reports").mkdir(exist_ok=True)

ratings, movies, _ = load_data()
train, test = train_test_split(ratings, test_size=0.2, random_state=42)
X_train, X_test = make_features(train, test, movies)
results = []

# ---------- Regression ----------
y_train, y_test = train["rating"], test["rating"]
reg_models = {
    "baseline_global_mean": DummyRegressor(strategy="mean"),
    "linear_regression": LinearRegression(),
    "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
}
for name, model in reg_models.items():
    model.fit(X_train, y_train)
    for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        pred = model.predict(X)
        results.append({
            "task": "regression", "model": name, "split": split,
            "rmse": np.sqrt(mean_squared_error(y, pred)),
            "mae": mean_absolute_error(y, pred),
        })

# ---------- Classification ----------
yc_train = (train["rating"] >= 4).astype(int)
yc_test = (test["rating"] >= 4).astype(int)
print("share of 'liked' in train:", round(yc_train.mean(), 3))

clf_models = {
    "baseline_most_frequent": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)),
    "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
}
for name, model in clf_models.items():
    model.fit(X_train, yc_train)
    for split, X, y in [("train", X_train, yc_train), ("test", X_test, yc_test)]:
        pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        results.append({
            "task": "classification", "model": name, "split": split,
            "accuracy": accuracy_score(y, pred),
            "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, proba),
        })

df = pd.DataFrame(results)
df.to_csv("reports/results_reg_clf.csv", index=False)
print(df.round(4).to_string(index=False))