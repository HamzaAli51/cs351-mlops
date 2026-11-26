import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from src.data.features import FeatureEncoder
from src.data.ingest import load_data
from src.models.recommender import SVDRecommender

MODELS_DIR = Path("models")
LIKE_THRESHOLD = 4.0


def split_data(ratings, test_size=0.2, seed=42):
    return train_test_split(ratings, test_size=test_size, random_state=seed)


def engineer_features(train, movies):
    encoder = FeatureEncoder(movies)
    X_train = encoder.fit_transform(train)
    return encoder, X_train


def train_models(encoder, X_train, train, movies):
    y = train["rating"]
    regressor = HistGradientBoostingRegressor(random_state=42).fit(X_train, y)
    classifier = HistGradientBoostingClassifier(random_state=42).fit(
        X_train, (y >= LIKE_THRESHOLD).astype(int))
    recommender = SVDRecommender(n_factors=10).fit(train)
    return {"encoder": encoder, "regressor": regressor,
            "classifier": classifier, "recommender": recommender,
            "titles": movies.set_index("movieId")["title"].to_dict()}


def evaluate(models, test):
    X_test = models["encoder"].transform(test)
    y_reg = test["rating"]
    y_clf = (y_reg >= LIKE_THRESHOLD).astype(int)

    reg_pred = models["regressor"].predict(X_test)
    clf_pred = models["classifier"].predict(X_test)
    clf_proba = models["classifier"].predict_proba(X_test)[:, 1]
    svd_pred = models["recommender"].predict_df(test)

    return {
        "n_test_rows": int(len(test)),
        "reg_rmse": float(np.sqrt(mean_squared_error(y_reg, reg_pred))),
        "clf_accuracy": float(accuracy_score(y_clf, clf_pred)),
        "clf_f1": float(f1_score(y_clf, clf_pred)),
        "clf_roc_auc": float(roc_auc_score(y_clf, clf_proba)),
        "svd_rmse": float(np.sqrt(mean_squared_error(y_reg, svd_pred))),
    }


def save_bundle(models, metrics, root=MODELS_DIR):
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(root) / version
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, out_dir / "bundle.joblib")
    (out_dir / "metrics.json").write_text(
        json.dumps({"version": version, "metrics": metrics}, indent=2))
    (Path(root) / "latest.txt").write_text(version)
    return version


def load_latest_bundle(root=MODELS_DIR):
    version = (Path(root) / "latest.txt").read_text().strip()
    return joblib.load(Path(root) / version / "bundle.joblib"), version


if __name__ == "__main__":
    ratings, movies, _ = load_data()
    train, test = split_data(ratings)
    encoder, X_train = engineer_features(train, movies)
    models = train_models(encoder, X_train, train, movies)
    metrics = evaluate(models, test)
    print(json.dumps(metrics, indent=2))
    print("saved version:", save_bundle(models, metrics))
    