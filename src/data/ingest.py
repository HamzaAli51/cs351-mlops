from pathlib import Path
import io
import zipfile

import pandas as pd
import requests

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RAW_DIR = Path("data/raw")


def download_movielens() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "ml-latest-small"
    if target.exists():
        return target
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(RAW_DIR)
    return target


def load_data():
    folder = download_movielens()
    ratings = pd.read_csv(folder / "ratings.csv")
    movies = pd.read_csv(folder / "movies.csv")
    tags = pd.read_csv(folder / "tags.csv")
    return ratings, movies, tags


if __name__ == "__main__":
    ratings, movies, tags = load_data()
    print(ratings.shape, movies.shape, tags.shape)
    print(ratings.head())