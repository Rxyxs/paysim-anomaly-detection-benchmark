"""Módulo 10 — ¿El ranking de detectores de PaySim se sostiene en fraude real?

PaySim es un simulador: genera transacciones a partir de patrones agregados de un
operador móvil africano, y el fraude sigue una regla fija (vaciar la cuenta de origen).
Todo lo medido en los módulos 1 a 9 es, por lo tanto, una afirmación sobre ese
simulador. Este módulo corre **los mismos detectores, con el mismo protocolo** (30.000
normales para ajustar, 50.000 normales más todos los fraudes para evaluar, RobustScaler,
mismas semillas) sobre un dataset de fraude **real**:

- **Credit Card Fraud (ULB, Kaggle `mlg-ulb/creditcardfraud`)**: 284.807 transacciones
  reales con tarjeta de clientes europeos en dos días de septiembre de 2013, 492
  fraudes (0,17%). Las variables V1-V28 son componentes principales de los datos
  originales (anonimizados por el banco); se agrega log(1 + monto).

Como la prevalencia de prueba difiere entre datasets (14% en PaySim, 1% acá), el PR-AUC
no es comparable directamente. La comparación usa ROC-AUC, que no depende de la
prevalencia, y el *lift* del PR-AUC sobre la prevalencia. La pregunta es si el **orden**
de los detectores se mantiene.

Además se mide una variante temporal propia de ULB: ajustar con normales del primer
día y evaluar en el segundo, que es como se usaría el detector.

    python -m src.realdata.ulb_benchmark --csv data/raw/ulb/creditcard.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

from src.unsupervised.benchmark import MODEL_LABELS, evaluate_all, run_detectors  # noqa: E402
from src.unsupervised.ensemble import build_ensembles  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "data" / "processed" / "realdata_ulb.json"
FIGURE_PATH = ROOT / "data" / "processed" / "figures" / "realdata_paysim_vs_ulb.png"
TRAIN_NORMAL_SIZE = 30_000
TEST_NORMAL_SIZE = 50_000
SECONDS_PER_DAY = 86_400

# ROC-AUC y PR-AUC de PaySim del módulo 3 (tabla "Results" del README, corrida completa
# sobre 6,36 M filas, prueba con 14,1% de fraude). Se copian acá para no volver a
# descargar 6 M de filas solo para releer números ya medidos.
PAYSIM_MODULE3 = {
    "gmm_density": (0.948, 0.807), "lof": (0.932, 0.802), "deep_svdd": (0.860, 0.702),
    "robust_mahalanobis": (0.896, 0.698), "abod": (0.888, 0.682), "ensemble_rank_avg": (0.875, 0.626),
    "autoencoder": (0.850, 0.581), "knn_distance": (0.853, 0.552), "isolation_forest": (0.850, 0.549),
    "ocsvm_nystroem": (0.830, 0.494), "ensemble_z_avg": (0.837, 0.493), "pca_reconstruction": (0.822, 0.487),
    "hbos": (0.800, 0.480), "ensemble_z_max": (0.854, 0.472), "hs_trees": (0.788, 0.458),
    "vae": (0.731, 0.446), "loda": (0.776, 0.354), "ecod": (0.672, 0.273), "mad_baseline": (0.383, 0.140),
}
PAYSIM_TEST_PREVALENCE = 8_213 / (50_000 + 8_213)


def load_ulb(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["log_amount"] = np.log1p(df["Amount"])
    return df


FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount"]


def random_split(df: pd.DataFrame, seed: int = 42):
    normal, fraud = df[df["Class"] == 0], df[df["Class"] == 1]
    train = normal.sample(n=TRAIN_NORMAL_SIZE, random_state=seed)
    test_normal = normal.drop(train.index).sample(n=TEST_NORMAL_SIZE, random_state=seed)
    test = pd.concat([test_normal, fraud]).sample(frac=1, random_state=seed)
    return train[FEATURES], test[FEATURES], test["Class"]


def temporal_split(df: pd.DataFrame, seed: int = 42):
    """Ajusta con normales del día 1 y evalúa todo el día 2 a su prevalencia real."""
    day1, day2 = df[df["Time"] < SECONDS_PER_DAY], df[df["Time"] >= SECONDS_PER_DAY]
    train = day1[day1["Class"] == 0].sample(n=TRAIN_NORMAL_SIZE, random_state=seed)
    test = day2.sample(n=min(len(day2), 80_000), random_state=seed) if len(day2) > 80_000 else day2
    # Se conservan todos los fraudes del día 2 aunque se submuestreen normales.
    test = pd.concat([test[test["Class"] == 0], day2[day2["Class"] == 1]]).drop_duplicates()
    return train[FEATURES], test[FEATURES], test["Class"]


def run(X_train, X_test, y_test) -> dict[str, dict]:
    scaler = RobustScaler()
    outputs = run_detectors(scaler.fit_transform(X_train), scaler.transform(X_test))
    ensembles = build_ensembles({k: v["scores"] for k, v in outputs.items()})
    for name, scores in ensembles.items():
        outputs[name] = {"scores": scores, "fit_seconds": 0.0, "score_seconds": 0.0}
    results = evaluate_all(outputs, y_test)
    prevalence = float(y_test.mean())
    table = {}
    for name, r in results.items():
        p100 = r["precision_recall_at_k"].get(100, (None, None))[0]
        table[name] = {
            "label": MODEL_LABELS.get(name, name),
            "family": r["family"],
            "roc_auc": round(float(r["roc_auc"]), 3),
            "pr_auc": round(float(r["pr_auc"]), 3),
            "pr_lift": round(float(r["pr_auc"]) / prevalence, 1),
            "precision@100": None if p100 is None else round(float(p100), 2),
            "fit_s": round(float(r["fit_seconds"]), 2),
        }
    return {"prevalence": round(prevalence, 4), "n_test": int(len(y_test)), "n_fraud": int(y_test.sum()),
            "detectors": table}


def compare(ulb: dict) -> dict:
    names = [n for n in PAYSIM_MODULE3 if n in ulb["detectors"]]
    pay_roc = [PAYSIM_MODULE3[n][0] for n in names]
    ulb_roc = [ulb["detectors"][n]["roc_auc"] for n in names]
    rho, p = spearmanr(pay_roc, ulb_roc)
    pay_rank = pd.Series(pay_roc, index=names).rank(ascending=False)
    ulb_rank = pd.Series(ulb_roc, index=names).rank(ascending=False)
    return {"spearman_roc_rank": round(float(rho), 3), "p_value": round(float(p), 4),
            "rank_shift": {n: int(pay_rank[n] - ulb_rank[n]) for n in names}}


def plot(ulb: dict, path: Path = FIGURE_PATH) -> None:
    names = [n for n in PAYSIM_MODULE3 if n in ulb["detectors"] and not n.startswith("ensemble")]
    x = [PAYSIM_MODULE3[n][0] for n in names]
    y = [ulb["detectors"][n]["roc_auc"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0.3, 1], [0.3, 1], color="#c9c7c1", linewidth=1, zorder=1)
    ax.scatter(x, y, color="#2a78d6", zorder=3)
    for n, xi, yi in zip(names, x, y):
        ax.annotate(MODEL_LABELS.get(n, n), (xi, yi), textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xlabel("ROC-AUC en PaySim (simulado, módulo 3)")
    ax.set_ylabel("ROC-AUC en ULB (fraude real con tarjeta)")
    ax.set_title("Mismo detector, mismo protocolo: simulado vs. real", loc="left", fontweight="bold")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.grid(color="#eceae4")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "ulb" / "creditcard.csv")
    args = parser.parse_args()
    df = load_ulb(args.csv)
    print(f"ULB: {len(df):,} transacciones, {int(df['Class'].sum())} fraudes")

    start = time.perf_counter()
    print("\n== Split aleatorio (protocolo del módulo 3) ==")
    ulb_random = run(*random_split(df))
    print("\n== Split temporal (día 1 -> día 2) ==")
    ulb_temporal = run(*temporal_split(df))
    summary = {
        "dataset": "mlg-ulb/creditcardfraud",
        "random_split": ulb_random,
        "temporal_split": ulb_temporal,
        "paysim_test_prevalence": round(PAYSIM_TEST_PREVALENCE, 4),
        "comparison_random": compare(ulb_random),
        "comparison_temporal": compare(ulb_temporal),
        "elapsed_seconds": round(time.perf_counter() - start, 1),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot(ulb_random)
    rows = sorted(ulb_random["detectors"].items(), key=lambda kv: -kv[1]["roc_auc"])
    print(f"\n{'detector':<34}{'ULB ROC':>8}{'PaySim ROC':>11}{'ULB PR':>8}{'lift':>7}")
    for n, r in rows:
        pay = PAYSIM_MODULE3.get(n, (float('nan'),))[0]
        print(f"{r['label']:<34}{r['roc_auc']:>8.3f}{pay:>11.3f}{r['pr_auc']:>8.3f}{r['pr_lift']:>7.1f}")
    print("\nComparación:", summary["comparison_random"]["spearman_roc_rank"], summary["comparison_temporal"]["spearman_roc_rank"])


if __name__ == "__main__":
    main()
