import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class SVDRecommender:
    def __init__(self, n_factors: int = 50, min_ratings: int = 10):
        self.n_factors = n_factors
        self.min_ratings = min_ratings

    def fit(self, train: pd.DataFrame):
        self.global_mean = train["rating"].mean()
        self.user_mean = train.groupby("userId")["rating"].mean()
        self.item_ids = np.sort(train["movieId"].unique())
        self.users = {u: i for i, u in enumerate(np.sort(train["userId"].unique()).tolist())}
        self.items = {m: j for j, m in enumerate(self.item_ids.tolist())}
        self.item_count = (train["movieId"].value_counts()
                           .reindex(self.item_ids).to_numpy())
        self.seen = train.groupby("userId")["movieId"].apply(set).to_dict()

        rows = train["userId"].map(self.users).to_numpy()
        cols = train["movieId"].map(self.items).to_numpy()
        vals = (train["rating"] - train["userId"].map(self.user_mean)).to_numpy()
        matrix = csr_matrix((vals, (rows, cols)),
                            shape=(len(self.users), len(self.items)))

        self.svd = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self.user_factors = self.svd.fit_transform(matrix)
        self.item_factors = self.svd.components_
        return self

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        u = df["userId"].map(self.users)
        j = df["movieId"].map(self.items)
        pred = df["userId"].map(self.user_mean).fillna(self.global_mean).to_numpy(dtype=float, copy=True)
        known = (u.notna() & j.notna()).to_numpy()
        ui = u[known].astype(int).to_numpy()
        ji = j[known].astype(int).to_numpy()
        pred[known] += np.einsum("ij,ji->i",
                                 self.user_factors[ui], self.item_factors[:, ji])
        return np.clip(pred, 0.5, 5.0)

    def recommend(self, user_id: int, n: int = 10) -> list:
        if user_id not in self.users:
            return self.item_ids[np.argsort(self.item_count)[::-1][:n]].tolist()
        scores = self.user_factors[self.users[user_id]] @ self.item_factors
        scores[self.item_count < self.min_ratings] = -np.inf
        for m in self.seen.get(user_id, ()):
            scores[self.items[m]] = -np.inf
        return self.item_ids[np.argsort(scores)[::-1][:n]].tolist()

    def similar_items(self, movie_id: int, n: int = 10) -> list:
        j = self.items[movie_id]
        vecs = self.item_factors.T
        norms = np.linalg.norm(vecs, axis=1) + 1e-9
        sims = (vecs @ vecs[j]) / (norms * norms[j])
        sims[self.item_count < self.min_ratings] = -np.inf
        sims[j] = -np.inf
        return self.item_ids[np.argsort(sims)[::-1][:n]].tolist()