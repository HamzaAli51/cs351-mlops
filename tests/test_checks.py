import numpy as np
import pandas as pd
import pytest

from src.validation.checks import (assert_performance, check_data_integrity,
                                   check_drift, validate_data)

GOOD = {"reg_rmse": 0.86, "clf_roc_auc": 0.79, "svd_rmse": 0.92}


def problems(df, movies):
    return check_data_integrity(df, movies, min_rows=100)


def test_clean_data_passes(ratings, movies):
    assert problems(ratings, movies) == []


def test_duplicates_detected(ratings, movies):
    bad = pd.concat([ratings, ratings.iloc[[0]]], ignore_index=True)
    assert any("duplicate" in p for p in problems(bad, movies))


def test_out_of_range_detected(ratings, movies):
    bad = ratings.copy()
    bad.loc[0, "rating"] = 7.0
    assert any("range" in p for p in problems(bad, movies))


def test_bad_step_detected(ratings, movies):
    bad = ratings.copy()
    bad.loc[0, "rating"] = 3.3
    assert any("half-star" in p for p in problems(bad, movies))


def test_missing_values_detected(ratings, movies):
    bad = ratings.copy()
    bad.loc[0, "rating"] = np.nan
    assert any("missing" in p for p in problems(bad, movies))


def test_too_few_rows_detected(ratings, movies):
    assert any("rows" in p for p in check_data_integrity(ratings, movies, min_rows=10**6))


def test_validate_data_raises(ratings, movies):
    with pytest.raises(ValueError, match="Data validation failed"):
        validate_data(ratings.iloc[:5], movies, min_rows=100)


def test_no_drift_on_same_distribution():
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    b = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    assert not check_drift(a, b, ["x"])["drifted"].any()


def test_drift_detected_on_shift():
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    b = pd.DataFrame({"x": rng.normal(1.5, 1, 5000)})
    assert check_drift(a, b, ["x"])["drifted"].all()


def test_drift_detected_on_shifted_ratings(ratings):
    shifted = ratings.copy()
    shifted["rating"] = (shifted["rating"] - 1.0).clip(lower=0.5)
    assert check_drift(ratings, shifted, ["rating"])["drifted"].all()


def test_good_metrics_pass_gate():
    assert_performance(GOOD)


def test_bad_rmse_fails_gate():
    with pytest.raises(ValueError, match="regression RMSE"):
        assert_performance({**GOOD, "reg_rmse": 1.2})


def test_bad_auc_fails_gate():
    with pytest.raises(ValueError, match="classifier AUC"):
        assert_performance({**GOOD, "clf_roc_auc": 0.55})