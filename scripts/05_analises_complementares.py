# -*- coding: utf-8 -*-
"""Complementary analyses on the clock-corrected dataset (blocked CV, 1-h embargo).

  (a) replica ablation      - 1, 2 or 3 units per sensor type (all combinations)
  (b) paired significance   - moving-block bootstrap (2-h blocks, 5000 resamples)
                              of the MAE difference between models
  (c) dynamic error         - MAE in stable vs transient regime (upper quartile
                              of |d ref/dt|) and residual lag after calibration
  (d) inference footprint   - serialized size, parameters/nodes, latency

The TPOT-TS pipelines (seed 42) are included whenever script 04 has produced
them; rerun this script after the TPOT queue finishes.

Outputs: resultados/ablacao_replicas.csv, bootstrap_pares.csv,
         erro_dinamico.csv, custo_inferencia.csv, predicoes_oof.csv
"""
import itertools
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

import config as C

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(C.SEED)
BASE = {
    "linear": lambda: make_pipeline(MinMaxScaler(), LinearRegression()),
    "quadratic": lambda: make_pipeline(MinMaxScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression()),
    "RF": lambda: make_pipeline(MinMaxScaler(), RandomForestRegressor(
        300, min_samples_leaf=5, random_state=C.SEED, n_jobs=4)),
}


def tpot_factory(sensor):
    path = C.RES / "tpot" / f"corrigido_seed{C.SEED}_{sensor}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        pipe = pickle.load(fh)
    return lambda: clone(pipe)


def oof_predict(make, X, y):
    p = np.full(len(y), np.nan)
    for tr, te in C.blocked_folds(len(y)):
        p[te] = make().fit(X[tr], y[tr]).predict(X[te])
    return p


def block_bootstrap_ci(diff, block=24, n_boot=5000):
    n = len(diff)
    nb = int(np.ceil(n / block))
    starts = RNG.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = diff[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def best_lag(ref, x, max_lag=6):
    """Lag (in 5-min steps) maximising the correlation of first differences;
    positive = x lags the reference."""
    dr, dx = np.diff(ref), np.diff(x)
    sc = {}
    for k in range(-max_lag, max_lag + 1):
        a, b = (dr[:len(dr) - k], dx[k:]) if k >= 0 else (dr[-k:], dx[:k])
        sc[k] = np.corrcoef(a, b)[0, 1]
    return max(sc, key=sc.get) * 5


def main():
    df = pd.read_csv(C.DATA_CORRECTED, index_col=0, parse_dates=True)
    abl, boot, dyn, inf, oof_all = [], [], [], [], {"timestamp": df.index}

    for sensor, target in C.PAIRS:
        y = df[target].to_numpy(float)
        models = dict(BASE)
        tp = tpot_factory(sensor)
        if tp is not None:
            models["TPOT-TS"] = tp

        # (a) ablation
        for k in (1, 2, 3):
            for name, make in models.items():
                if name == "TPOT-TS" and k < 3:             # architecture was searched for 3 inputs
                    continue
                mae = [np.abs(y - oof_predict(make, df[C.feature_cols(sensor, combo)].to_numpy(float), y)).mean()
                       for combo in itertools.combinations(C.REPLICAS, k)]
                abl.append({"sensor": sensor, "replicas": k, "model": name,
                            "MAE_mean": np.mean(mae), "MAE_min": np.min(mae), "MAE_max": np.max(mae)})

        # out-of-fold predictions with the 3 replicas
        X = df[C.feature_cols(sensor)].to_numpy(float)
        pred = {name: oof_predict(make, X, y) for name, make in models.items()}
        oof_all[f"{sensor}__ref"] = y
        oof_all[f"{sensor}__raw"] = X.mean(axis=1)
        for name, p in pred.items():
            oof_all[f"{sensor}__{name}"] = p

        # (b) paired block bootstrap
        for a, b in itertools.combinations(models, 2):
            d = np.abs(y - pred[a]) - np.abs(y - pred[b])
            lo, hi = block_bootstrap_ci(d)
            boot.append({"sensor": sensor, "model_a": a, "model_b": b, "dMAE_a_minus_b": d.mean(),
                         "ci_lo": lo, "ci_hi": hi,
                         "verdict": f"{b} better" if lo > 0 else (f"{a} better" if hi < 0 else "equivalent")})

        # (c) dynamic error and residual lag
        rate = np.abs(np.gradient(y))
        transient = rate > np.quantile(rate, 0.75)
        for name, p in {"raw": X.mean(axis=1), **pred}.items():
            e = np.abs(y - p)
            dyn.append({"sensor": sensor, "model": name, "MAE_global": e.mean(),
                        "MAE_stable": e[~transient].mean(), "MAE_transient": e[transient].mean(),
                        "ratio": e[transient].mean() / e[~transient].mean(),
                        "residual_lag_min": best_lag(y, p)})

        # (d) inference footprint (model fitted on the full window)
        for name, make in models.items():
            m = make().fit(X, y)
            x1 = X[:1]
            t0 = time.perf_counter()
            for _ in range(200):
                m.predict(x1)
            last = m.steps[-1][1]
            size = getattr(last, "coef_", None)
            n_par = (int(np.size(last.coef_)) + 1 if size is not None else
                     int(sum(t.tree_.node_count for t in last.estimators_)) if hasattr(last, "estimators_")
                     and hasattr(last.estimators_[0], "tree_") else None)
            inf.append({"sensor": sensor, "model": name, "size_kB": len(pickle.dumps(m)) / 1024,
                        "parameters_or_nodes": n_par,
                        "latency_ms": (time.perf_counter() - t0) / 200 * 1e3})
        print(sensor, "done | models:", list(models), flush=True)

    fmt = dict(index=False, float_format="%.4f")
    pd.DataFrame(abl).to_csv(C.RES / "ablacao_replicas.csv", **fmt)
    pd.DataFrame(boot).to_csv(C.RES / "bootstrap_pares.csv", **fmt)
    pd.DataFrame(dyn).to_csv(C.RES / "erro_dinamico.csv", **fmt)
    pd.DataFrame(inf).to_csv(C.RES / "custo_inferencia.csv", **fmt)
    pd.DataFrame(oof_all).to_csv(C.RES / "predicoes_oof.csv", **fmt)

    pd.set_option("display.width", 220)
    print("\n(a) ablation - blocked-CV MAE\n", pd.DataFrame(abl).pivot_table(
        index="sensor", columns=["model", "replicas"], values="MAE_mean").round(2).to_string())
    print("\n(b) bootstrap\n", pd.DataFrame(boot).round(3).to_string(index=False))
    print("\n(c) dynamic error\n", pd.DataFrame(dyn).round(2).to_string(index=False))
    print("\n(d) inference\n", pd.DataFrame(inf).groupby("model")[["size_kB", "parameters_or_nodes", "latency_ms"]]
          .median().round(3).to_string())


if __name__ == "__main__":
    main()
