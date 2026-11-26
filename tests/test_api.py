import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture()
def client(bundle_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(bundle_dir))
    with TestClient(app) as c:
        yield c


def post_csv(client, name, content, mime="text/csv"):
    return client.post("/predict/batch", files={"file": (name, content, mime)})


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_model_info(client):
    assert "reg_rmse" in client.get("/model/info").json()["test_metrics"]


def test_predict_rating(client):
    r = client.post("/predict/rating", json={"user_id": 1, "movie_id": 1})
    assert r.status_code == 200
    body = r.json()
    assert 0.5 <= body["predicted_rating"] <= 5.0
    assert 0.0 <= body["liked_probability"] <= 1.0
    assert body["title"] == "Movie 1 (2000)"


def test_predict_rating_rejects_bad_input(client):
    assert client.post("/predict/rating", json={"user_id": 0, "movie_id": 1}).status_code == 422
    assert client.post("/predict/rating", json={"user_id": 1}).status_code == 422


def test_unknown_user_and_movie_still_predict(client):
    r = client.post("/predict/rating", json={"user_id": 999999, "movie_id": 999999})
    assert r.status_code == 200
    assert r.json()["title"] is None


def test_features_endpoint_reports_unknown_genres(client):
    r = client.post("/predict/features", json={
        "user_mean": 4.0, "user_count": 20, "movie_mean": 4.0,
        "movie_count": 30, "genres": ["Drama", "NotAGenre"]})
    assert r.status_code == 200
    assert r.json()["unknown_genres"] == ["NotAGenre"]


def test_features_endpoint_rejects_out_of_range(client):
    r = client.post("/predict/features", json={
        "user_mean": 9.0, "user_count": 20, "movie_mean": 4.0, "movie_count": 30})
    assert r.status_code == 422


def test_recommend(client):
    body = client.get("/recommend/1?n=5").json()
    assert body["known_user"] is True
    assert len(body["recommendations"]) == 5


def test_recommend_rejects_bad_n(client):
    assert client.get("/recommend/1?n=0").status_code == 422
    assert client.get("/recommend/1?n=500").status_code == 422


def test_batch_upload(client):
    r = post_csv(client, "pairs.csv", b"userId,movieId\n1,1\n2,2\n999999,1\n")
    assert r.status_code == 200
    assert r.json()["n_rows"] == 3


def test_batch_rejects_wrong_file_type(client):
    assert post_csv(client, "pairs.txt", b"hello", "text/plain").status_code == 400


def test_batch_rejects_missing_columns(client):
    assert post_csv(client, "pairs.csv", b"user,movie\n1,1\n").status_code == 422