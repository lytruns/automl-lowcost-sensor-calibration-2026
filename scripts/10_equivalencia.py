# -*- coding: utf-8 -*-
"""Equivalence test (TOST logic) of the quadratic model against RF and TPOT-TS,
on the blocked-CV out-of-fold absolute errors of the clock-corrected dataset.

Two models are declared practically equivalent when the 90 % moving-block
bootstrap CI (4-h blocks, conservative) of the MAE difference lies entirely
inside +/- delta.  Margins are metrological, fixed a priori:
  temperature: 0.2 degC (HMP45C accuracy at 20 degC) and 0.5 degC (value quoted in the manuscript)
  humidity   : 2 % RH   (HMP45C accuracy, 0-90 % RH)

Output: resultados/equivalencia.csv
"""
import numpy as np
import pandas as pd

import config as C

RNG = np.random.default_rng(C.SEED)
MARGINS = {"temp": (0.2, 0.5), "umid": (2.0,)}


def mbb_ci(d, block=48, n_boot=5000, level=90):
    n = len(d); nb = int(np.ceil(n / block))
    st = RNG.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (st[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    a = (100 - level) / 2
    return np.percentile(d[idx].mean(axis=1), [a, 100 - a])


def main():
    oof = pd.read_csv(C.RES / "predicoes_oof.csv")
    rows = []
    for other in ("RF", "TPOT-TS"):
        for sensor, target in C.PAIRS:
            y = oof[f"{sensor}__ref"].to_numpy()
            d = np.abs(y - oof[f"{sensor}__quadratic"].to_numpy()) - np.abs(y - oof[f"{sensor}__{other}"].to_numpy())
            lo, hi = mbb_ci(d)
            row = {"comparison": f"quadratic - {other}", "sensor": sensor, "dMAE": d.mean(), "ci90_lo": lo, "ci90_hi": hi}
            for m in MARGINS[target]:
                row[f"equiv_within_{m}"] = bool(lo > -m and hi < m)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(C.RES / "equivalencia.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
