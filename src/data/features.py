import pandas as pd
from sklearn.model_selection import KFold

K = 5          # smoothing strength
N_FOLDS = 5    # folds for out-of-fold encoding


def genre_matrix(movies: pd.DataFrame) -> pd.DataFrame:
    genres = movies["genres"].str.get_dummies(sep="|")
    genres.index = movies["movieId"]
    return genres.drop(columns=["(no genres listed)"], errors="ignore")


class FeatureEncoder:
    def __init__(self, movies: pd.DataFrame):
        self.genres = genre_matrix(movies)

    @staticmethod
    def _stats(source: pd.DataFrame):
        u = source.groupby("userId")["rating"].agg(["sum", "count"])
        m = source.groupby("movieId")["rating"].agg(["sum", "count"])
        return u, m

    def _encode(self, u, m, target: pd.DataFrame) -> pd.DataFrame:
        us = target["userId"].map(u["sum"]).fillna(0)
        uc = target["userId"].map(u["count"]).fillna(0)
        ms = target["movieId"].map(m["sum"]).fillna(0)
        mc = target["movieId"].map(m["count"]).fillna(0)

        out = pd.DataFrame(index=target.index)
        out["user_mean"] = (us + K * self.global_mean) / (uc + K)
        out["user_count"] = uc
        out["movie_mean"] = (ms + K * self.global_mean) / (mc + K)
        out["movie_count"] = mc
        return out

    def _add_genres(self, enc: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        g = self.genres.reindex(df["movieId"]).fillna(0)
        g.index = df.index
        return pd.concat([enc, g], axis=1)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        self.global_mean = train["rating"].mean()
        self.user_stats, self.movie_stats = self._stats(train)

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        parts = []
        for fit_idx, enc_idx in kf.split(train):
            u, m = self._stats(train.iloc[fit_idx])
            parts.append(self._encode(u, m, train.iloc[enc_idx]))
        enc = pd.concat(parts).loc[train.index]
        return self._add_genres(enc, train)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        enc = self._encode(self.user_stats, self.movie_stats, df)
        scale = (N_FOLDS - 1) / N_FOLDS
        enc[["user_count", "movie_count"]] *= scale
        return self._add_genres(enc, df)


def make_features(train: pd.DataFrame, test: pd.DataFrame, movies: pd.DataFrame):
    encoder = FeatureEncoder(movies)
    return encoder.fit_transform(train), encoder.transform(test)