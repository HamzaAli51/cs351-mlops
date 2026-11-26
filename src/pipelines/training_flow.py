import time

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from src.data.ingest import load_data
from src.models.train import (engineer_features, evaluate, save_bundle,
                              split_data, train_models)
from src.pipelines.notify import notify

MIN_ROWS = 50_000
MAX_REG_RMSE = 0.90
MIN_CLF_AUC = 0.75
MAX_SVD_RMSE = 0.95


@task(name="ingest-data", retries=3, retry_delay_seconds=[5, 15, 30],
      cache_policy=NO_CACHE)
def ingest_task():
    ratings, movies, _ = load_data()
    print(f"loaded {len(ratings)} ratings and {len(movies)} movies")
    return ratings, movies


@task(name="validate-data", cache_policy=NO_CACHE)
def validate_task(ratings, movies):
    problems = []
    if len(ratings) < MIN_ROWS:
        problems.append(f"only {len(ratings)} rating rows (need {MIN_ROWS}+)")
    if ratings[["userId", "movieId", "rating"]].isna().any().any():
        problems.append("missing values in ratings")
    if not ratings["rating"].between(0.5, 5.0).all():
        problems.append("ratings outside the 0.5-5.0 range")
    if ratings.duplicated(subset=["userId", "movieId"]).any():
        problems.append("duplicate user-movie pairs")
    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))
    print("data validation passed")


@task(name="split-data", cache_policy=NO_CACHE)
def split_task(ratings):
    train, test = split_data(ratings)
    print(f"train rows: {len(train)}, test rows: {len(test)}")
    return train, test


@task(name="engineer-features", cache_policy=NO_CACHE)
def feature_task(train, movies):
    return engineer_features(train, movies)


@task(name="train-models", cache_policy=NO_CACHE)
def train_task(encoder, X_train, train, movies):
    return train_models(encoder, X_train, train, movies)


@task(name="evaluate-models", cache_policy=NO_CACHE)
def evaluate_task(models, test):
    metrics = evaluate(models, test)
    print(metrics)
    return metrics


@task(name="quality-gate", cache_policy=NO_CACHE)
def gate_task(metrics):
    failures = []
    if metrics["reg_rmse"] > MAX_REG_RMSE:
        failures.append(f"regression RMSE {metrics['reg_rmse']:.4f} > {MAX_REG_RMSE}")
    if metrics["clf_roc_auc"] < MIN_CLF_AUC:
        failures.append(f"classifier AUC {metrics['clf_roc_auc']:.4f} < {MIN_CLF_AUC}")
    if metrics["svd_rmse"] > MAX_SVD_RMSE:
        failures.append(f"SVD RMSE {metrics['svd_rmse']:.4f} > {MAX_SVD_RMSE}")
    if failures:
        raise ValueError("Quality gate failed: " + "; ".join(failures))
    print("quality gate passed")


@task(name="save-model", retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE)
def save_task(models, metrics):
    version = save_bundle(models, metrics)
    print("saved model version", version)
    return version


@flow(name="movielens-training-pipeline", log_prints=True)
def training_pipeline():
    start = time.perf_counter()
    try:
        ratings, movies = ingest_task()
        validate_task(ratings, movies)
        train, test = split_task(ratings)
        encoder, X_train = feature_task(train, movies)
        models = train_task(encoder, X_train, train, movies)
        metrics = evaluate_task(models, test)
        gate_task(metrics)
        version = save_task(models, metrics)
    except Exception as exc:
        notify(f"❌ Training pipeline FAILED: {type(exc).__name__}: {exc}")
        raise

    seconds = time.perf_counter() - start
    notify(
        f"✅ Training pipeline succeeded\n"
        f"Model version: {version}\n"
        f"Regression RMSE: {metrics['reg_rmse']:.4f} | "
        f"Classifier AUC: {metrics['clf_roc_auc']:.4f} | "
        f"SVD RMSE: {metrics['svd_rmse']:.4f}\n"
        f"Duration: {seconds:.0f}s"
    )
    return {"version": version, "metrics": metrics}


if __name__ == "__main__":
    training_pipeline()