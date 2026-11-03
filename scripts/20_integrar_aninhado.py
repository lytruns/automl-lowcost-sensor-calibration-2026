# -*- coding: utf-8 -*-
"""Integrate the nested TPOT evaluation (script 18) into the result files used by the manuscript.

  * predicoes_oof.csv : column <sensor>__TPOT-TS  <- nested out-of-fold predictions
                        (the former fixed-architecture predictions are kept as <sensor>__TPOT-fixed)
  * tpot_protocolos.csv (dataset 'corrigido'):
        blocked  <- nested metrics (mean and sd over the four outer blocks)
        holdout  kept (the search never saw the final 20 %)
        random / walkfwd rows removed: the architecture had been selected on data that overlap
        their test sets, so they are not reported.
  * tpot_aninhado_resumo.csv : one row per nested search (pipeline, generations, test metrics)
Idempotent: can be run again without duplicating anything.
"""
import json
import re

import numpy as np
import pandas as pd

import config as C

NEST = C.RES / "tpot_aninhado"


def main():
    oof = pd.read_csv(C.RES / "predicoes_oof.csv")
    prot = pd.read_csv(C.RES / "tpot_protocolos.csv")
    rows, summary = [], []
    for s, target in C.PAIRS:
        y = oof[f"{s}__ref"].to_numpy()
        pred = np.full(len(y), np.nan)
        fold_metrics = []
        for k in range(C.N_BLOCKS):
            d = json.loads((NEST / f"{s}_fold{k}.json").read_text(encoding="utf-8"))
            a, b = d["test_index"]
            pred[a:b + 1] = d["pred"]
            fold_metrics.append(C.metrics(y[a:b + 1], np.asarray(d["pred"])))
            steps = [e for e in re.findall(r"([A-Z][A-Za-z]+)\(", d["pipeline"])
                     if e not in {"Pipeline", "FeatureUnion", "Passthrough", "SkipTransformer", "FunctionTransformer"}]
            summary.append({"sensor": s, "fold": k, "n_generations": d["n_generations"], "n_evaluated": d["n_evaluated"],
                            "duration_s": d["duration_s"], "MAE_test": d["MAE_test"], "R2_test": d["R2_test"],
                            "final_estimator": steps[-1] if steps else "", "pipeline": " > ".join(dict.fromkeys(steps))})
        assert not np.isnan(pred).any(), s
        if f"{s}__TPOT-fixed" not in oof.columns:
            oof[f"{s}__TPOT-fixed"] = oof[f"{s}__TPOT-TS"]
        oof[f"{s}__TPOT-TS"] = pred
        m = pd.DataFrame(fold_metrics)
        rows.append({"dataset": "corrigido", "sensor": s, "target": target, "model": "TPOT-TS", "protocol": "blocked",
                     "n_folds": len(m), **{f"{c}_mean": m[c].mean() for c in m}, **{f"{c}_sd": m[c].std(ddof=0) for c in m}})
        old = prot[(prot.dataset == "corrigido") & (prot.sensor == s) & (prot.protocol == "blocked")].MAE_mean
        print(f"{s:9s} blocked-CV MAE  fixed architecture = {np.abs(y - oof[f'{s}__TPOT-fixed']).mean():.3f}   nested = {m.MAE.mean():.3f}")

    keep = prot[~((prot.dataset == "corrigido") & prot.protocol.isin(["blocked", "random", "walkfwd"]))]
    prot_new = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
    prot_new.to_csv(C.RES / "tpot_protocolos.csv", index=False, float_format="%.4f")
    oof.to_csv(C.RES / "predicoes_oof.csv", index=False, float_format="%.4f")
    summ = pd.DataFrame(summary)
    summ.to_csv(C.RES / "tpot_aninhado_resumo.csv", index=False, float_format="%.4f")
    print("\nfinal estimators selected in the 32 nested searches:\n", summ.final_estimator.value_counts().to_string())
    print("\nsearch duration (s): median %.0f, range %.0f-%.0f" % (summ.duration_s.median(), summ.duration_s.min(), summ.duration_s.max()))


if __name__ == "__main__":
    main()
