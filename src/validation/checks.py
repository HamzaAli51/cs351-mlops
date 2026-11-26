import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

MIN_ROWS = 50_000
PSI_LIMIT = 0.25
THRESHOLDS = {
    "reg_rmse_max": 0.90,
    "clf_roc_auc_min": 0.75,
    "svd_rmse_max": 0.95,
}


def check_data_integrity(ratings, movies, min_rows=MIN_ROWS):
    required = ["userId", "movieId", "rating"]
    missing_cols = [c for c in required if c not in ratings.columns]
    if missing_cols:
        return [f"missing columns: {missing_cols}"]

    problems = []
    if len(ratings) < min_rows:
        problems.append(f"only {len(ratings)} rating rows (need {min_rows}+)")
    if ratings[required].isna().any().any():
        problems.append("missing values in ratings")
    if not ratings["rating"].between(0.5, 5.0).all():
        problems.append("ratings outside the 0.5-5.0 range")
    elif not ((ratings["rating"] * 2) % 1 == 0).all():
        problems.append("ratings not in half-star steps")
    if ratings.duplicated(subset=["userId", "movieId"]).any():
        problems.append("duplicate user-movie pairs")
    orphan_share = (~ratings["movieId"].isin(movies["movieId"])).mean()
    if orphan_share > 0.01:
        problems.append(f"{orphan_share:.1%} of ratings reference unknown movies")
    return problems


def validate_data(ratings, movies, min_rows=MIN_ROWS):
    problems = check_data_integrity(ratings, movies, min_rows)
    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))


def psi(reference, current, bins=10):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(np.unique(reference)) <= 20:
        cats = np.union1d(np.unique(reference), np.unique(current))
        ref_p = np.array([(reference == c).mean() for c in cats])
        cur_p = np.array([(current == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_p = np.histogram(reference, bins=edges)[0] / len(reference)
        cur_p = np.histogram(current, bins=edges)[0] / len(current)
    ref_p = np.clip(ref_p, 1e-4, None)
    cur_p = np.clip(cur_p, 1e-4, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def check_drift(reference, current, columns, psi_limit=PSI_LIMIT):
    rows = []
    for col in columns:
        ks = ks_2samp(reference[col], current[col])
        score = psi(reference[col], current[col])
        rows.append({"column": col, "ks_statistic": float(ks.statistic),
                     "ks_pvalue": float(ks.pvalue), "psi": score,
                     "drifted": score > psi_limit})
    return pd.DataFrame(rows)


def check_performance(metrics, thresholds=THRESHOLDS):
    failures = []
    if metrics["reg_rmse"] > thresholds["reg_rmse_max"]:
        failures.append(f"regression RMSE {metrics['reg_rmse']:.4f} "
                        f"> {thresholds['reg_rmse_max']}")
    if metrics["clf_roc_auc"] < thresholds["clf_roc_auc_min"]:
        failures.append(f"classifier AUC {metrics['clf_roc_auc']:.4f} "
                        f"< {thresholds['clf_roc_auc_min']}")
    if metrics["svd_rmse"] > thresholds["svd_rmse_max"]:
        failures.append(f"SVD RMSE {metrics['svd_rmse']:.4f} "
                        f"> {thresholds['svd_rmse_max']}")
    return failures


def assert_performance(metrics, thresholds=THRESHOLDS):
    failures = check_performance(metrics, thresholds)
    if failures:
        raise ValueError("Quality gate failed: " + "; ".join(failures))