import pandas as pd

K = 5  # smoothing strength


def genre_matrix(movies: pd.DataFrame) -> pd.DataFrame:
    genres = movies["genres"].str.get_dummies(sep="|")
    genres.index = movies["movieId"]
    return genres.drop(columns=["(no genres listed)"], errors="ignore")


def make_features(train: pd.DataFrame, test: pd.DataFrame, movies: pd.DataFrame):
    global_mean = train["rating"].mean()
    genres = genre_matrix(movies)

    u_sum = train.groupby("userId")["rating"].sum()
    u_cnt = train.groupby("userId")["rating"].count()
    m_sum = train.groupby("movieId")["rating"].sum()
    m_cnt = train.groupby("movieId")["rating"].count()

    def build(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        us = df["userId"].map(u_sum).fillna(0)
        uc = df["userId"].map(u_cnt).fillna(0)
        ms = df["movieId"].map(m_sum).fillna(0)
        mc = df["movieId"].map(m_cnt).fillna(0)

        if is_train:
            us, uc = us - df["rating"], uc - 1
            ms, mc = ms - df["rating"], mc - 1

        out = pd.DataFrame(index=df.index)
        out["user_mean"] = (us + K * global_mean) / (uc + K)
        out["user_count"] = uc
        out["movie_mean"] = (ms + K * global_mean) / (mc + K)
        out["movie_count"] = mc

        g = genres.reindex(df["movieId"]).fillna(0)
        g.index = df.index
        return pd.concat([out, g], axis=1)

    return build(train, True), build(test, False)