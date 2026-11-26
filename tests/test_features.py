import numpy as np
import pandas as pd
import pytest

from src.data.features import FeatureEncoder, genre_matrix

COLS = ["user_mean", "user_count", "movie_mean", "movie_count"]


def test_training_features_do_not_leak_own_label(ratings, movies):
    train = ratings.reset_index(drop=True)
    changed = train.copy()
    changed.loc[0, "rating"] = 0.5 if train.loc[0, "rating"] > 2.5 else 5.0

    f1 = FeatureEncoder(movies).fit_transform(train)
    f2 = FeatureEncoder(movies).fit_transform(changed)

    np.testing.assert_allclose(f1.loc[0, COLS].astype(float),
                               f2.loc[0, COLS].astype(float), atol=0.01)


def test_unknown_user_and_movie_fall_back_to_global_mean(ratings, movies):
    enc = FeatureEncoder(movies)
    enc.fit_transform(ratings)
    out = enc.transform(pd.DataFrame({"userId": [999999], "movieId": [999999]}))
    assert out.loc[0, "user_count"] == 0
    assert out.loc[0, "movie_count"] == 0
    assert out.loc[0, "user_mean"] == pytest.approx(enc.global_mean)
    assert out.loc[0, "movie_mean"] == pytest.approx(enc.global_mean)


def test_genre_matrix_drops_unlisted():
    movies = pd.DataFrame({"movieId": [1, 2], "title": ["a", "b"],
                           "genres": ["Drama|Comedy", "(no genres listed)"]})
    g = genre_matrix(movies)
    assert "(no genres listed)" not in g.columns
    assert g.loc[1, "Drama"] == 1
    assert g.loc[2].sum() == 0