from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

from src.data.ingest import load_data

TOP_N = 200
MIN_SUPPORT = 0.12
MIN_LIFT = 1.2
MIN_CONFIDENCE = 0.6

Path("reports").mkdir(exist_ok=True)

ratings, movies, _ = load_data()
titles = movies.set_index("movieId")["title"]

liked = ratings[ratings["rating"] >= 4]
top_movies = liked["movieId"].value_counts().head(TOP_N).index
liked = liked[liked["movieId"].isin(top_movies)]

basket = pd.crosstab(liked["userId"], liked["movieId"]).gt(0)
print("baskets (users):", basket.shape[0], "items (movies):", basket.shape[1])

frequent = fpgrowth(basket, min_support=MIN_SUPPORT, use_colnames=True, max_len=3)
print("frequent itemsets:", len(frequent))

try:
    rules = association_rules(frequent, num_itemsets=len(basket),
                              metric="lift", min_threshold=MIN_LIFT)
except TypeError:
    rules = association_rules(frequent, metric="lift", min_threshold=MIN_LIFT)

rules = rules[rules["confidence"] >= MIN_CONFIDENCE]
rules = rules.sort_values("lift", ascending=False)
print("rules:", len(rules))


def names(items):
    return " + ".join(titles[m] for m in sorted(items))


out = pd.DataFrame({
    "if_liked": rules["antecedents"].apply(names),
    "then_also_liked": rules["consequents"].apply(names),
    "support": rules["support"],
    "confidence": rules["confidence"],
    "lift": rules["lift"],
})
out.to_csv("reports/results_association_rules.csv", index=False)
print(out.head(10).round(3).to_string(index=False))