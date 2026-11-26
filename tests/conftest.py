import numpy as np
import pandas as pd
import pytest

from src.models.train import (engineer_features, evaluate, save_bundle,
                              split_data, train_models)

GENRES = ["Drama", "Comedy", "Action", "Thriller", "Romance"]


@pytest.fixture(scope="session")
def movies():
    rng = np.random.default_rng(0)
    rows = []
    for movie_id in range(1, 41):
        picked = rng.choice(GENRES, size=int(rng.integers(1, 3)), replace=False)
        rows.append({"movieId": movie_id, "title": f"Movie {movie_id} (2000)",
                     "genres": "|".join(picked)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def ratings():
    rng = np.random.default_rng(1)
    user_bias = rng.normal(0, 0.5, size=60)
    movie_bias = rng.normal(0, 0.7, size=40)
    rows = []
    for user in range(60):
        for movie in rng.choice(40, size=25, replace=False):
            raw = 3.5 + user_bias[user] + movie_bias[movie] + rng.normal(0, 0.5)
            rating = float(np.clip(np.round(raw * 2) / 2, 0.5, 5.0))
            rows.append({"userId": user + 1, "movieId": int(movie) + 1,
                         "rating": rating, "timestamp": 1_000_000_000 + len(rows)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory, ratings, movies):
    train, test = split_data(ratings)
    encoder, X_train = engineer_features(train, movies)
    models = train_models(encoder, X_train, train, movies)
    metrics = evaluate(models, test)
    root = tmp_path_factory.mktemp("models")
    save_bundle(models, metrics, root=root)
    return root