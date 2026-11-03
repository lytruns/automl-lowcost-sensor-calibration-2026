# -*- coding: utf-8 -*-
"""Polynomial degree study on the clock-corrected dataset.

For degrees 1-5: in-sample MAE (fit and evaluate on the same data) versus
out-of-sample MAE (blocked CV and walk-forward), with the 3 replicas as inputs
and with their mean as a single input. Also the worst single-block MAE, which
exposes edge-of-range oscillation.  Output: resultados/grau_polinomio.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

import config as C

warnings.filterwarnings("ignore")


def model(deg):
    return make_pipeline(MinMaxScaler(), PolynomialFeatures(deg, include_bias=False), LinearRegression())


def main():
    df = pd.read_csv(C.DATA_CORRECTED, index_col=0, parse_dates=True)
    rows = []
    for sensor, target in C.PAIRS:
        y = df[target].to_numpy(float)
        X3 = df[C.feature_cols(sensor)].to_numpy(float)
        for inputs, X in (("3 replicas", X3), ("mean of replicas", X3.mean(axis=1, keepdims=True))):
            for deg in range(1, 6):
                n_coef = model(deg).fit(X, y).steps[-1][1].coef_.size + 1
                insample = np.abs(y - model(deg).fit(X, y).predict(X)).mean()
                blk = [np.abs(y[te] - model(deg).fit(X[tr], y[tr]).predict(X[te])).mean()
                       for tr, te in C.blocked_folds(len(y))]
                wf = [np.abs(y[te] - model(deg).fit(X[tr], y[tr]).predict(X[te])).mean()
                      for tr, te in C.walk_forward_folds(len(y))]
                rows.append({"sensor": sensor, "target": target, "inputs": inputs, "degree": deg,
                             "n_coef": n_coef, "MAE_insample": insample, "MAE_blocked": np.mean(blk),
                             "MAE_worst_block": np.max(blk), "MAE_walkfwd": np.mean(wf)})
    out = pd.DataFrame(rows)
    out.to_csv(C.RES / "grau_polinomio.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    for inputs in ("3 replicas", "mean of replicas"):
        for target, name in (("temp", "TEMPERATURE (mean of 5 sensors, degC)"), ("umid", "HUMIDITY (mean of 3 sensors, %RH)")):
            t = out[(out.inputs == inputs) & (out.target == target)].groupby("degree")[
                ["n_coef", "MAE_insample", "MAE_blocked", "MAE_worst_block", "MAE_walkfwd"]].mean()
            print(f"\n{name} | inputs: {inputs}\n{t.round(2).to_string()}")


if __name__ == "__main__":
    main()
