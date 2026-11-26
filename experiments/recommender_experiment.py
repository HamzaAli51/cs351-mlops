from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.data.ingest import load_data
from src.models.recommender import SVDRecommender

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

ratings, movies, _ = load_data()
train, test = train_test_split(ratings, test_size=0.2, random_state=42)
global_mean = train["rating"].mean()


def rmse(pred):
    return float(np.sqrt(mean_squared_error(test["rating"], pred)))


def precision_at_k(rec_fn, k=10):
    relevant = test[test["rating"] >= 4].groupby("userId")["movieId"].apply(set)
    scores = [len(set(rec_fn(user, k)) & rel) / k for user, rel in relevant.items()]
    return float(np.mean(scores))


popular = train["movieId"].value_counts().index.tolist()
seen = train.groupby("userId")["movieId"].apply(set)


def popularity_recs(user, k):
    s = seen[user]
    return [m for m in popular if m not in s][:k]


results = [
    {"model": "baseline_global_mean", "rmse": rmse(np.full(len(test), global_mean)),
     "precision_at_10": np.nan},
    {"model": "baseline_user_mean",
     "rmse": rmse(test["userId"].map(train.groupby("userId")["rating"].mean())
                  .fillna(global_mean)),
     "precision_at_10": np.nan},
    {"model": "baseline_popularity", "rmse": np.nan,
     "precision_at_10": precision_at_k(popularity_recs)},
]

fitted = {}
for k in [10, 20, 50, 100]:
    model = SVDRecommender(n_factors=k).fit(train)
    fitted[k] = model
    results.append({
        "model": f"svd_{k}_factors",
        "rmse": rmse(model.predict_df(test)),
        "precision_at_10": precision_at_k(model.recommend),
    })

df = pd.DataFrame(results)
df.to_csv("reports/results_recommender.csv", index=False)
print(df.round(4).to_string(index=False))

svd_rows = df[df["model"].str.startswith("svd")]
best_k = int(svd_rows.loc[svd_rows["rmse"].idxmin(), "model"].split("_")[1])
best = fitted[best_k]
joblib.dump(best, "models/recommender.joblib")
print("\nbest n_factors by RMSE:", best_k)

titles = movies.set_index("movieId")["title"]
print("\nTop 10 for user 1:")
print(titles.loc[best.recommend(1, 10)].to_string())
print("\nMovies similar to Toy Story (movieId 1):")
print(titles.loc[best.similar_items(1, 10)].to_string())