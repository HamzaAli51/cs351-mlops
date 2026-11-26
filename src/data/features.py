import pandas as pd
from sklearn.model_selection import KFold

K = 5          # smoothing strength
N_FOLDS = 5    # folds for out-of-fold encoding


def genre_matrix(movies: pd.DataFrame) -> pd.DataFrame:
    genres = movies["genres"].str.get_dummies(sep="|")
    genres.index = movies["movieId"]
    return genres.drop(columns=["(no genres listed)"], errors="ignore")


def _encode(source: pd.DataFrame, target: pd.DataFrame, global_mean: float):
    u = source.groupby("userId")["rating"].agg(["sum", "count"])
    m = source.groupby("movieId")["rating"].agg(["sum", "count"])

    us = target["userId"].map(u["sum"]).fillna(0)
    uc = target["userId"].map(u["count"]).fillna(0)
    ms = target["movieId"].map(m["sum"]).fillna(0)
    mc = target["movieId"].map(m["count"]).fillna(0)

    out = pd.DataFrame(index=target.index)
    out["user_mean"] = (us + K * global_mean) / (uc + K)
    out["user_count"] = uc
    out["movie_mean"] = (ms + K * global_mean) / (mc + K)
    out["movie_count"] = mc
    return out


def make_features(train: pd.DataFrame, test: pd.DataFrame, movies: pd.DataFrame):
    global_mean = train["rating"].mean()
    genres = genre_matrix(movies)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    parts = []
    for fit_idx, enc_idx in kf.split(train):
        parts.append(_encode(train.iloc[fit_idx], train.iloc[enc_idx], global_mean))
    train_enc = pd.concat(parts).loc[train.index]

    test_enc = _encode(train, test, global_mean)
    scale = (N_FOLDS - 1) / N_FOLDS
    test_enc[["user_count", "movie_count"]] *= scale

    def add_genres(enc: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        g = genres.reindex(df["movieId"]).fillna(0)
        g.index = df.index
        return pd.concat([enc, g], axis=1)

    return add_genres(train_enc, train), add_genres(test_enc, test)