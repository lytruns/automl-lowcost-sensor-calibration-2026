# -*- coding: utf-8 -*-
"""Closing protocol for temporal support and dynamic response (clock-converted data).

The CR23X stored air temperature with P71 (Average, 5 min) and relative humidity
with P70 (Sample, value at the output instant).  Low-cost series are therefore
built with a support matched to each reference variable.

Step 1-2  RH under three supports: A = trailing 5-min mean, B = trailing 1-min
          mean, C = 10-s reading nearest to the reference stamp.  Same rows and
          same blocked-CV partitions; paired moving-block bootstrap CI (2-h
          blocks) of the MAE difference between supports.
Step 3    Dynamic response with matched supports (T: 5-min mean; RH: B and C):
          (a) fine delay relative to the 60-min time-base conversion, all data /
              day (06-18 h) / night, mean +/- sd over the three units;
          (b) residual of the calibrated quadratic model (out-of-fold) regressed
              on the rate of change (tau: rate-DEPENDENT lag, min) and on its
              sign (h: rate-INDEPENDENT offset between rising and falling
              branches, i.e. hysteresis in the strict sense), block-bootstrap CIs;
          (c) blocked-CV MAE with a dynamic term (sensor's own rate of change).

Outputs: resultados/ur_suporte_temporal.csv, resultados/resposta_dinamica.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

import config as C

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(C.SEED)
SUPPORT = {"A_5min": 300, "B_1min": 60, "C_10s": 10}
MODELS = {
    "linear": lambda: make_pipeline(MinMaxScaler(), LinearRegression()),
    "quadratic": lambda: make_pipeline(MinMaxScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression()),
    "RF": lambda: make_pipeline(MinMaxScaler(), RandomForestRegressor(300, min_samples_leaf=5, random_state=C.SEED, n_jobs=8)),
}


def at_stamps(series, stamps, offset_s, win_s):
    n = max(1, win_s // C.GRID_S)
    r = series.rolling(n, min_periods=max(1, n // 2)).mean()
    r.index = r.index - pd.Timedelta(seconds=int(offset_s))
    return r.reindex(stamps)


def oof(make, X, y):
    p = np.full(len(y), np.nan)
    for tr, te in C.blocked_folds(len(y)):
        p[te] = make().fit(X[tr], y[tr]).predict(X[te])
    return p


def mbb(stat, n, block=24, n_boot=3000):
    nb = int(np.ceil(n / block)); out = []
    for _ in range(n_boot):
        st = RNG.integers(0, n - block + 1, nb)
        out.append(stat((st[:, None] + np.arange(block)).ravel()[:n]))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


def main():
    df = pd.read_csv(C.DATA_CORRECTED, index_col=0, parse_dates=True)
    dev = {d: pd.read_pickle(C.PROC / f"_grid10s_device_{d}.pkl") for d in C.REPLICAS}
    stamps = df.index

    # ---------------- Step 1-2: RH temporal support ----------------
    rows = []
    for sensor in C.RH_SENSORS:
        Xs = {k: pd.concat([at_stamps(dev[d][sensor], stamps, C.CLOCK_OFFSET_S, w) for d in C.REPLICAS], axis=1)
              for k, w in SUPPORT.items()}
        ok = np.logical_and.reduce([x.notna().all(axis=1).to_numpy() for x in Xs.values()])
        y = df["umid"].to_numpy(float)[ok]
        for name, make in MODELS.items():
            err = {k: np.abs(y - oof(make, x.to_numpy(float)[ok], y)) for k, x in Xs.items()}
            row = {"sensor": sensor, "model": name, "n": int(ok.sum()), **{f"MAE_{k}": e.mean() for k, e in err.items()}}
            for k in ("B_1min", "C_10s"):
                d = err["A_5min"] - err[k]
                lo, hi = mbb(lambda i: d[i].mean(), len(d))
                row.update({f"dMAE_A_minus_{k}": d.mean(), f"ci_lo_{k}": lo, f"ci_hi_{k}": hi})
            rows.append(row)
    sup = pd.DataFrame(rows)
    sup.to_csv(C.RES / "ur_suporte_temporal.csv", index=False, float_format="%.4f")

    # ---------------- Step 3: dynamic response ----------------
    hour = stamps.hour
    periods = {"all": np.ones(len(stamps), bool), "day": (hour >= 6) & (hour < 18), "night": (hour < 6) | (hour >= 18)}
    shifts = np.arange(-900, 901, 10)
    rows = []
    cases = [(s, "temp", "A_5min") for s in C.TEMP_SENSORS] + [(s, "umid", k) for s in C.RH_SENSORS for k in ("B_1min", "C_10s")]
    for sensor, target, sup_key in cases:
        win = SUPPORT[sup_key]
        dref = df[target].diff()
        row = {"sensor": sensor, "support": sup_key}
        # (a) fine delay by period
        for pname, mask in periods.items():
            lags = []
            for d in C.REPLICAS:
                sc = []
                for s in shifts:
                    a = at_stamps(dev[d][sensor], stamps, C.CLOCK_OFFSET_S + s, win).diff()
                    ok = a.notna().to_numpy() & dref.notna().to_numpy() & mask
                    sc.append(np.corrcoef(a.to_numpy()[ok], dref.to_numpy()[ok])[0, 1])
                lags.append(shifts[int(np.argmax(sc))] / 60)
            row[f"delay_{pname}"] = np.mean(lags); row[f"delay_{pname}_sd"] = np.std(lags)
        # (b) residual ~ rate + sign(rate)
        X = pd.concat([at_stamps(dev[d][sensor], stamps, C.CLOCK_OFFSET_S, win) for d in C.REPLICAS], axis=1)
        ok = X.notna().all(axis=1).to_numpy()
        Xv, y = X.to_numpy(float)[ok], df[target].to_numpy(float)[ok]
        res = y - oof(MODELS["quadratic"], Xv, y)
        rate = np.gradient(y) / 5.0
        sgn = np.where(np.abs(rate) > np.quantile(np.abs(rate), 0.25), np.sign(rate), 0.0)
        A = np.column_stack([np.ones_like(rate), rate, sgn])
        fit = lambda i: np.linalg.lstsq(A[i], res[i], rcond=None)[0][1:]
        tau, h = fit(np.arange(len(y)))
        (tlo, hlo), (thi, hhi) = mbb(fit, len(y))
        # (c) dynamic compensation
        dX = np.vstack([np.zeros((1, 3)), np.diff(Xv, axis=0)]).mean(axis=1, keepdims=True)
        res_dyn = y - oof(MODELS["quadratic"], np.hstack([Xv, dX]), y)
        fast = np.abs(rate) > np.quantile(np.abs(rate), 0.75)
        row.update({"tau_min": tau, "tau_lo": tlo, "tau_hi": thi, "h_units": h, "h_lo": hlo, "h_hi": hhi,
                    "MAE_quad": np.abs(res).mean(), "MAE_quad_dyn": np.abs(res_dyn).mean(),
                    "MAE_transient": np.abs(res)[fast].mean(), "MAE_transient_dyn": np.abs(res_dyn)[fast].mean()})
        rows.append(row)
        print(sensor, sup_key, "done", flush=True)
    dyn = pd.DataFrame(rows)
    dyn.to_csv(C.RES / "resposta_dinamica.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 250)
    print("\nRH TEMPORAL SUPPORT (blocked-CV MAE, % RH)\n", sup.round(3).to_string(index=False))
    print("\nDYNAMIC RESPONSE\n", dyn.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
