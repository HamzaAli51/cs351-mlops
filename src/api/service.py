import json
import logging
import os
from pathlib import Path

import pandas as pd

from src.models.train import MODELS_DIR, load_latest_bundle

log = logging.getLogger("mlops.api")


class ModelService:
    def __init__(self, root=None):
        root = Path(root or os.getenv("MODEL_DIR", MODELS_DIR))
        self.models, self.version = load_latest_bundle(root)
        info = json.loads((root / self.version / "metrics.json").read_text())
        self.metrics = info["metrics"]
        self.encoder = self.models["encoder"]
        self.titles = self.models["titles"]
        self.feature_cols = list(self.models["regressor"].feature_names_in_)
        self.known_genres = set(self.encoder.genres.columns)
        log.info("loaded model version %s", self.version)

    def predict_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.encoder.transform(df)[self.feature_cols]
        out = df[["userId", "movieId"]].copy()
        out["predicted_rating"] = self.models["regressor"].predict(X).clip(0.5, 5.0)
        out["liked_probability"] = self.models["classifier"].predict_proba(X)[:, 1]
        return out

    def predict_features(self, req) -> dict:
        row = {c: 0.0 for c in self.feature_cols}
        row.update(user_mean=req.user_mean, user_count=req.user_count,
                   movie_mean=req.movie_mean, movie_count=req.movie_count)
        unknown = []
        for genre in req.genres:
            if genre in self.known_genres:
                row[genre] = 1.0
            else:
                unknown.append(genre)

        X = pd.DataFrame([row])[self.feature_cols]
        rating = float(self.models["regressor"].predict(X).clip(0.5, 5.0)[0])
        prob = float(self.models["classifier"].predict_proba(X)[0, 1])
        return {
            "predicted_rating": round(rating, 3),
            "liked_probability": round(prob, 4),
            "liked": prob >= 0.5,
            "unknown_genres": unknown,
            "model_version": self.version,
        }

    def recommend(self, user_id: int, n: int):
        recommender = self.models["recommender"]
        ids = recommender.recommend(user_id, n)
        known = user_id in recommender.users
        items = [(int(m), self.titles.get(int(m), "unknown")) for m in ids]
        return known, items