# -*- coding: utf-8 -*-
"""Heating-cooling offset h (rate-independent) and rate term tau for the out-of-fold
residuals of EVERY model family (linear, quadratic, RF, TPOT-TS), to check whether the
asymmetry found for the quadratic model is specific to it.  Same regression and
bootstrap as script 13.  Output: resultados/assimetria_por_modelo.csv"""
import numpy as np
import pandas as pd

import config as C

RNG = np.random.default_rng(C.SEED)


def mbb(stat, n, block=24, n_boot=3000):
    nb = int(np.ceil(n / block)); out = []
    for _ in range(n_boot):
        st = RNG.integers(0, n - block + 1, nb)
        out.append(stat((st[:, None] + np.arange(block)).ravel()[:n]))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


def main():
    oof = pd.read_csv(C.RES / "predicoes_oof.csv")
    rows = []
    for s, _ in C.PAIRS:
        y = oof[f"{s}__ref"].to_numpy()
        rate = np.gradient(y) / 5.0
        sgn = np.where(np.abs(rate) > np.quantile(np.abs(rate), 0.25), np.sign(rate), 0.0)
        A = np.column_stack([np.ones_like(rate), rate, sgn])
        for m in ("linear", "quadratic", "RF", "TPOT-TS"):
            res = y - oof[f"{s}__{m}"].to_numpy()
            fit = lambda i: np.linalg.lstsq(A[i], res[i], rcond=None)[0][1:]
            tau, h = fit(np.arange(len(y)))
            (tlo, hlo), (thi, hhi) = mbb(fit, len(y))
            rows.append({"sensor": s, "model": m, "h": h, "h_lo": hlo, "h_hi": hhi, "h_sig": bool(hlo > 0 or hhi < 0),
                         "tau": tau, "tau_lo": tlo, "tau_hi": thi})
    out = pd.DataFrame(rows)
    out.to_csv(C.RES / "assimetria_por_modelo.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    t = out.assign(cell=out.apply(lambda r: f"{r.h:+.2f}{'*' if r.h_sig else ' '}", axis=1)).pivot(index="sensor", columns="model", values="cell")
    print("Heating-cooling offset h by model (* = 95% CI excludes zero)\n", t[["linear", "quadratic", "RF", "TPOT-TS"]].to_string())
    print("\nsignificant h, by model:", out.groupby("model").h_sig.sum().to_dict())


if __name__ == "__main__":
    main()
