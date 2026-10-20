"""Módulo 10, segundo dataset real: IEEE-CIS (Vesta, e-commerce con tarjeta no presente).

Mismo protocolo que `ulb_benchmark.py` (30.000 normales para ajustar, 50.000 normales más
todos los fraudes de la muestra para evaluar, RobustScaler, mismos detectores), sobre
`train_transaction.csv` de la competencia de Kaggle `ieee-fraud-detection`: 590.540
transacciones reales, 3,5% de fraude. A diferencia de ULB, acá las variables no son
componentes principales, así que este dataset no favorece por construcción a los modelos
lineales y gaussianos.

Variables: log(1 + monto), los contadores C1-C14 y los deltas de tiempo D1, D4, D10 y D15,
con faltantes reemplazados por la mediana de entrenamiento más un indicador de faltante.
Las ~340 columnas V de Vesta se dejan fuera: son derivadas de las anteriores y llenas de
faltantes. Como hay 20.663 fraudes, la prueba usa una muestra de 5.000 para quedar cerca de
la prevalencia de ULB (~9%).

    python -m src.realdata.ieee_benchmark --zip data/raw/ieee/ieee-fraud-detection.zip
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.realdata.ulb_benchmark import PAYSIM_MODULE3, TEST_NORMAL_SIZE, TRAIN_NORMAL_SIZE, compare, run

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "data" / "processed" / "realdata_ieee.json"
TEST_FRAUD_SIZE = 5_000
SECONDS_PER_DAY = 86_400
BASE_COLUMNS = [f"C{i}" for i in range(1, 15)] + ["D1", "D4", "D10", "D15"]


def load_ieee(zip_path: Path) -> pd.DataFrame:
    usecols = ["TransactionDT", "TransactionAmt", "isFraud"] + BASE_COLUMNS
    with zipfile.ZipFile(zip_path) as z, z.open("train_transaction.csv") as f:
        return pd.read_csv(f, usecols=usecols)


def features(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame({"log_amount": np.log1p(df["TransactionAmt"])})
    for col in BASE_COLUMNS:
        X[f"{col}_missing"] = df[col].isna().astype(float) if col.startswith("D") else 0.0
        X[col] = df[col].fillna(medians[col])
    return X.loc[:, (X != 0).any(axis=0)]


def split(df: pd.DataFrame, temporal: bool, seed: int = 42):
    if temporal:
        cut = df["TransactionDT"].quantile(0.7)
        past, future = df[df["TransactionDT"] < cut], df[df["TransactionDT"] >= cut]
    else:
        past = future = df
    train = past[past["isFraud"] == 0].sample(n=TRAIN_NORMAL_SIZE, random_state=seed)
    pool = future.drop(train.index, errors="ignore")
    test_normal = pool[pool["isFraud"] == 0].sample(n=TEST_NORMAL_SIZE, random_state=seed)
    frauds = pool[pool["isFraud"] == 1]
    test_fraud = frauds.sample(n=min(TEST_FRAUD_SIZE, len(frauds)), random_state=seed)
    test = pd.concat([test_normal, test_fraud]).sample(frac=1, random_state=seed)
    medians = train[BASE_COLUMNS].median()
    X_train, X_test = features(train, medians), features(test, medians)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    return X_train, X_test, test["isFraud"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, default=ROOT / "data" / "raw" / "ieee" / "ieee-fraud-detection.zip")
    args = parser.parse_args()
    df = load_ieee(args.zip)
    print(f"IEEE-CIS: {len(df):,} transacciones, {int(df['isFraud'].sum()):,} fraudes")
    summary = {"dataset": "ieee-fraud-detection (train_transaction.csv)"}
    for name, temporal in (("random_split", False), ("temporal_split", True)):
        print(f"\n== {name} ==")
        summary[name] = run(*split(df, temporal))
        summary[f"comparison_{name.split('_')[0]}"] = compare(summary[name])
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = sorted(summary["random_split"]["detectors"].items(), key=lambda kv: -kv[1]["pr_auc"])
    for n, r in rows:
        t = summary["temporal_split"]["detectors"][n]
        print(f"{r['label']:<34}{PAYSIM_MODULE3.get(n, (float('nan'),))[0]:>7.3f}{r['roc_auc']:>7.3f}{r['pr_auc']:>7.3f}{t['pr_auc']:>7.3f}")
    print("Spearman ROC vs PaySim:", summary["comparison_random"]["spearman_roc_rank"], summary["comparison_temporal"]["spearman_roc_rank"])


if __name__ == "__main__":
    main()
