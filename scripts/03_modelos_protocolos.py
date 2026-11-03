# -*- coding: utf-8 -*-
"""Reference models under every validation protocol, on both datasets.

Models   : raw (mean of replicas, no calibration), linear, quadratic,
           random forest (RF), gradient boosting (HGB).
Protocols: random   - shuffled 80/20 split, 10 repetitions (temporal leakage)
           holdout  - temporal 80/20 (primary metric of the manuscript)
           blocked  - 4 contiguous blocks (~1 day), 1-h embargo
           walkfwd  - expanding window, 5 folds, 1-h embargo (past -> future)
Every model is a Pipeline whose MinMaxScaler is refitted on each training set.

Output: resultados/modelos_protocolos.csv (long format, one row per
        dataset x sensor x model x protocol, mean and sd across folds/repeats)
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

import config as C

warnings.filterwarnings("ignore")

MODELS = {
    "linear": lambda: make_pipeline(MinMaxScaler(), LinearRegression()),
    "quadratic": lambda: make_pipeline(MinMaxScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression()),
    "RF": lambda: make_pipeline(MinMaxScaler(), RandomForestRegressor(
        300, min_samples_leaf=5, random_state=C.SEED, n_jobs=4)),
    "HGB": lambda: make_pipeline(MinMaxScaler(), HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, random_state=C.SEED)),
}


def protocols(n):
    yield "random", list(ShuffleSplit(10, test_size=C.TEST_SIZE, random_state=C.SEED).split(np.arange(n)))
    yield "holdout", [C.holdout_index(n)]
    yield "blocked", list(C.blocked_folds(n))
    yield "walkfwd", list(C.walk_forward_folds(n))


def evaluate(make, X, y, folds):
    out = [C.metrics(y[te], make().fit(X[tr], y[tr]).predict(X[te])) for tr, te in folds]
    return pd.DataFrame(out)


def main():
    rows = []
    for tag, path in [("original", C.DATA_ORIGINAL), ("corrigido", C.DATA_CORRECTED)]:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        for sensor, target in C.PAIRS:
            X, y = df[C.feature_cols(sensor)].to_numpy(float), df[target].to_numpy(float)
            for proto, folds in protocols(len(y)):
                raw = pd.DataFrame([C.metrics(y[te], X[te].mean(axis=1)) for _, te in folds])
                res = {"raw": raw, **{k: evaluate(mk, X, y, folds) for k, mk in MODELS.items()}}
                for model, m in res.items():
                    rows.append({"dataset": tag, "sensor": sensor, "target": target,
                                 "model": model, "protocol": proto, "n_folds": len(m),
                                 **{f"{c}_mean": m[c].mean() for c in m},
                                 **{f"{c}_sd": m[c].std(ddof=0) for c in m}})
            print(tag, sensor, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(C.RES / "modelos_protocolos.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 250)
    for tag in ("original", "corrigido"):
        for metric in ("R2_mean", "MAE_mean"):
            t = out[(out.dataset == tag) & (out.model != "raw")].pivot_table(
                index="sensor", columns=["model", "protocol"], values=metric)
            t = t.reindex(columns=["linear", "quadratic", "RF", "HGB"], level=0)
            print(f"\n===== {tag.upper()} | {metric} =====\n{t.round(2).to_string()}")


if __name__ == "__main__":
    main()
