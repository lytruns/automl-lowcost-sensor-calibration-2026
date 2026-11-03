# -*- coding: utf-8 -*-
"""Robustness of the paired model comparisons (blocked-CV out-of-fold errors,
clock-corrected dataset):

  (a) moving-block bootstrap 95 % CI of the MAE difference for block lengths of
      1, 2, 4 and 6 h (sensitivity to the block-length choice);
  (b) Diebold-Mariano test on the absolute-error loss differential with a
      Newey-West (HAC) variance, truncation lag = 2 h, two-sided;
  (c) Holm correction of the DM p-values across the 8 sensor/variable pairs.

Output: resultados/testes_robustez.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

import config as C

RNG = np.random.default_rng(C.SEED)
PAIRS_M = [("linear", "quadratic"), ("quadratic", "RF"), ("quadratic", "TPOT-TS"), ("RF", "TPOT-TS")]


def mbb_ci(d, block, n_boot=5000):
    n = len(d); nb = int(np.ceil(n / block))
    st = RNG.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (st[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    return np.percentile(d[idx].mean(axis=1), [2.5, 97.5])


def diebold_mariano(d, lag):
    n = len(d); dm = d - d.mean()
    var = dm @ dm / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dm[k:] @ dm[:-k]) / n
    stat = d.mean() / np.sqrt(var / n)
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))


def holm(p):
    p = np.asarray(p); order = np.argsort(p); adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i]); adj[i] = min(1.0, running)
    return adj


def main():
    oof = pd.read_csv(C.RES / "predicoes_oof.csv")
    rows = []
    for a, b in PAIRS_M:
        for sensor, _ in C.PAIRS:
            y = oof[f"{sensor}__ref"].to_numpy()
            d = np.abs(y - oof[f"{sensor}__{a}"].to_numpy()) - np.abs(y - oof[f"{sensor}__{b}"].to_numpy())
            stat, p = diebold_mariano(d, lag=24)
            row = {"model_a": a, "model_b": b, "sensor": sensor, "dMAE_a_minus_b": d.mean(), "DM_stat": stat, "DM_p": p}
            for hours in (1, 2, 4, 6):
                lo, hi = mbb_ci(d, block=12 * hours)
                row[f"sig_{hours}h"] = bool(lo > 0 or hi < 0)
            rows.append(row)
    out = pd.DataFrame(rows)
    out["DM_p_holm"] = out.groupby(["model_a", "model_b"]).DM_p.transform(lambda p: holm(p.to_numpy()))
    out["DM_sig_holm"] = out.DM_p_holm < 0.05
    out.to_csv(C.RES / "testes_robustez.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    print("\nSignificant differences out of 8 sensors, by criterion:")
    print(out.groupby(["model_a", "model_b"])[["sig_1h", "sig_2h", "sig_4h", "sig_6h", "DM_sig_holm"]].sum().to_string())


if __name__ == "__main__":
    main()
