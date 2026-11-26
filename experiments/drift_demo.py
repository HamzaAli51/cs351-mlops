from sklearn.model_selection import train_test_split

from src.data.ingest import load_data
from src.validation.checks import check_drift

ratings, _, _ = load_data()
cols = ["rating"]

train, test = train_test_split(ratings, test_size=0.2, random_state=42)
print("1) Random split (expect no drift):")
print(check_drift(train, test, cols).round(4).to_string(index=False))

ordered = ratings.sort_values("timestamp")
cut = int(len(ordered) * 0.8)
early, late = ordered.iloc[:cut], ordered.iloc[cut:]
print("\n2) Temporal split, oldest 80% vs newest 20%:")
print(check_drift(early, late, cols).round(4).to_string(index=False))

broken = test.copy()
broken["rating"] = (broken["rating"] - 1.0).clip(lower=0.5)
print("\n3) Injected fault, every rating shifted down one star:")
print(check_drift(train, broken, cols).round(4).to_string(index=False))