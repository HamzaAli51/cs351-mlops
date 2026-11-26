from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data.features import genre_matrix
from src.data.ingest import load_data

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)

ratings, movies, _ = load_data()

# ---- 1. One taste profile per user ----
genres = genre_matrix(movies)
g = genres.reindex(ratings["movieId"]).fillna(0)
g.index = ratings.index
g["userId"] = ratings["userId"]
profile = g.groupby("userId").mean()
profile["avg_rating"] = ratings.groupby("userId")["rating"].mean()
genre_cols = [c for c in profile.columns if c != "avg_rating"]
print("profile shape:", profile.shape)

scaler = StandardScaler()
X = scaler.fit_transform(profile)

# ---- 2. PCA ----
pca_full = PCA(random_state=42).fit(X)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n90 = int(np.argmax(cum >= 0.90)) + 1
print(f"components needed for 90% variance: {n90} of {X.shape[1]}")
print("variance explained by PC1, PC2:",
      pca_full.explained_variance_ratio_[:2].round(3))

plt.plot(range(1, len(cum) + 1), cum, marker="o")
plt.axhline(0.90, linestyle="--")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA scree plot")
plt.savefig(FIG_DIR / "pca_variance.png", bbox_inches="tight")
plt.close()

pca2 = PCA(n_components=2, random_state=42)
coords = pca2.fit_transform(X)
loadings = pd.Series(pca2.components_[0], index=profile.columns).sort_values()
print("\nPC1 most negative:", loadings.head(3).index.tolist())
print("PC1 most positive:", loadings.tail(3).index.tolist())

# ---- 3. Choose k for K-Means ----
rows = []
for k in range(3, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"k": k, "inertia": km.inertia_,
                 "silhouette": silhouette_score(X, km.labels_)})
scores = pd.DataFrame(rows)
scores.to_csv("reports/results_clustering.csv", index=False)
print("\n", scores.round(4).to_string(index=False))
best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
print("best k by silhouette:", best_k)

# ---- 4. Final clustering and interpretation ----
km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
profile["cluster"] = km.labels_
overall = profile[genre_cols].mean()
for c in range(best_k):
    members = profile[profile["cluster"] == c]
    diff = (members[genre_cols].mean() - overall).sort_values(ascending=False)
    print(f"\nCluster {c}: {len(members)} users, "
          f"avg rating {members['avg_rating'].mean():.2f}")
    print("  over-represented:", ", ".join(diff.head(3).index))
    print("  under-represented:", ", ".join(diff.tail(3).index))

plt.scatter(coords[:, 0], coords[:, 1], c=km.labels_, cmap="tab10", s=15)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"User taste clusters (k={best_k}) in PCA space")
plt.savefig(FIG_DIR / "user_clusters.png", bbox_inches="tight")
plt.close()

joblib.dump({"scaler": scaler, "kmeans": km, "columns": list(profile.columns[:-1])},
            "models/clustering.joblib")