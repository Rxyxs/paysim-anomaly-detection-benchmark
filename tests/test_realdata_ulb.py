"""Tests del módulo 10 (datos reales ULB) con un DataFrame sintético con el esquema de ULB."""
import numpy as np
import pandas as pd

from src.realdata import ulb_benchmark as ub


def _fake_ulb(n=200_000, n_fraud=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = rng.exponential(50, n)
    df["Time"] = np.sort(rng.uniform(0, 2 * ub.SECONDS_PER_DAY, n))
    df["Class"] = 0
    df.loc[rng.choice(n, n_fraud, replace=False), "Class"] = 1
    df["log_amount"] = np.log1p(df["Amount"])
    return df


def test_random_split_trains_only_on_normals_and_keeps_every_fraud():
    df = _fake_ulb()
    X_train, X_test, y_test = ub.random_split(df)
    assert len(X_train) == ub.TRAIN_NORMAL_SIZE
    assert int(y_test.sum()) == int(df["Class"].sum())
    assert int((y_test == 0).sum()) == ub.TEST_NORMAL_SIZE
    assert list(X_train.columns) == ub.FEATURES
    assert X_train.index.intersection(X_test.index).empty


def test_temporal_split_fits_on_day_one_and_scores_day_two():
    df = _fake_ulb()
    X_train, X_test, y_test = ub.temporal_split(df)
    assert (df.loc[X_train.index, "Time"] < ub.SECONDS_PER_DAY).all()
    assert (df.loc[X_test.index, "Time"] >= ub.SECONDS_PER_DAY).all()
    day2_frauds = int(df.loc[(df["Time"] >= ub.SECONDS_PER_DAY) & (df["Class"] == 1)].shape[0])
    assert int(y_test.sum()) == day2_frauds


def test_compare_reports_rank_correlation_against_paysim():
    fake = {"detectors": {name: {"roc_auc": roc} for name, (roc, _) in ub.PAYSIM_MODULE3.items()}}
    result = ub.compare(fake)
    assert result["spearman_roc_rank"] == 1.0
    assert all(shift == 0 for shift in result["rank_shift"].values())


def test_ieee_split_fits_on_normals_and_fills_missing_from_train():
    from src.realdata import ieee_benchmark as ib

    rng = np.random.default_rng(1)
    n = 300_000
    df = pd.DataFrame({c: rng.poisson(3, n).astype(float) for c in ib.BASE_COLUMNS})
    df.loc[rng.choice(n, n // 3, replace=False), "D4"] = np.nan
    df["TransactionAmt"] = rng.exponential(80, n)
    df["TransactionDT"] = np.arange(n)
    df["isFraud"] = (rng.random(n) < 0.05).astype(int)
    X_train, X_test, y_test = ib.split(df, temporal=True)
    assert len(X_train) == ub.TRAIN_NORMAL_SIZE
    assert not X_train.isna().any().any() and not X_test.isna().any().any()
    assert "D4_missing" in X_train.columns
    assert list(X_test.columns) == list(X_train.columns)
    future_frauds = int(df.loc[df["TransactionDT"] >= df["TransactionDT"].quantile(0.7), "isFraud"].sum())
    assert int(y_test.sum()) == min(ib.TEST_FRAUD_SIZE, future_frauds)
