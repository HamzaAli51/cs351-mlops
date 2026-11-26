from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.ingest import load_data

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

ratings, movies, tags = load_data()

print("=== Shapes ===")
print(ratings.shape, movies.shape, tags.shape)
print(ratings.describe())

print("\n=== Data quality ===")
print("missing values in ratings:\n", ratings.isna().sum())
print("duplicate rows:", ratings.duplicated().sum())
print("duplicate user-movie pairs:",
      ratings.duplicated(subset=["userId", "movieId"]).sum())
print("movies with no genre:",
      (movies["genres"] == "(no genres listed)").sum())

ratings["rating"].value_counts().sort_index().plot(kind="bar")
plt.title("Rating distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.savefig(FIG_DIR / "rating_distribution.png", bbox_inches="tight")
plt.close()

print("\n=== Sparsity ===")
n_users = ratings["userId"].nunique()
n_movies = ratings["movieId"].nunique()
density = len(ratings) / (n_users * n_movies)
print("users:", n_users, "movies:", n_movies)
print(f"density: {density:.4f}, sparsity: {1 - density:.4f}")

print("\n=== Long tail ===")
per_user = ratings.groupby("userId").size()
per_movie = ratings.groupby("movieId").size()
print(per_user.describe())
print(per_movie.describe())
print("movies with only 1 rating:", (per_movie == 1).sum())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
per_user.plot(kind="hist", bins=50, ax=ax[0], title="Ratings per user")
per_movie.plot(kind="hist", bins=50, ax=ax[1], title="Ratings per movie")
fig.savefig(FIG_DIR / "long_tail.png", bbox_inches="tight")
plt.close(fig)

print("\n=== Time range ===")
ratings["date"] = pd.to_datetime(ratings["timestamp"], unit="s")
print(ratings["date"].min(), "to", ratings["date"].max())
monthly = ratings.set_index("date").resample("ME")["rating"].count()
monthly.plot(title="Ratings per month")
plt.savefig(FIG_DIR / "ratings_per_month.png", bbox_inches="tight")
plt.close()

print("\n=== Genres ===")
genre_counts = movies["genres"].str.split("|").explode().value_counts()
print(genre_counts)
genre_counts.plot(kind="bar", figsize=(10, 4))
plt.savefig(FIG_DIR / "genre_counts.png", bbox_inches="tight")
plt.close()