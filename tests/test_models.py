import numpy as np

from src.models.recommender import SVDRecommender
from src.models.train import (engineer_features, evaluate, load_latest_bundle,
                              split_data, train_models)


def test_bundle_roundtrip(bundle_dir):
    models, version = load_latest_bundle(bundle_dir)
    assert {"encoder", "regressor", "classifier", "recommender", "titles"} <= set(models)
    assert (bundle_dir / version / "metrics.json").exists()


def test_models_beat_baseline(ratings, movies):
    train, test = split_data(ratings)
    encoder, X_train = engineer_features(train, movies)
    metrics = evaluate(train_models(encoder, X_train, train, movies), test)
    baseline_rmse = float(np.sqrt(((test["rating"] - train["rating"].mean()) ** 2).mean()))
    assert metrics["reg_rmse"] < baseline_rmse
    assert metrics["clf_roc_auc"] > 0.5


def test_svd_predictions_in_valid_range(ratings):
    train, test = split_data(ratings)
    pred = SVDRecommender(n_factors=5).fit(train).predict_df(test)
    assert len(pred) == len(test)
    assert pred.min() >= 0.5 and pred.max() <= 5.0


def test_recommend_excludes_seen_movies(ratings):
    train, _ = split_data(ratings)
    model = SVDRecommender(n_factors=5).fit(train)
    user = int(train["userId"].iloc[0])
    recs = model.recommend(user, 10)
    seen = set(train.loc[train["userId"] == user, "movieId"])
    assert len(recs) == 10
    assert not set(recs) & seen


def test_unknown_user_gets_fallback_recommendations(ratings):
    assert len(SVDRecommender(n_factors=5).fit(ratings).recommend(999999, 5)) == 5


def test_similar_items_excludes_itself(ratings):
    model = SVDRecommender(n_factors=5).fit(ratings)
    movie = int(ratings["movieId"].iloc[0])
    assert movie not in model.similar_items(movie, 5)