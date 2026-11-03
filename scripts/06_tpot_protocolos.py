# -*- coding: utf-8 -*-
"""Re-evaluate the TPOT-TS selected pipelines under every validation protocol
and summarise the stability of the evolutionary search across seeds.

The pipeline ARCHITECTURE selected by TPOT on the training portion (seed 42) is
refitted inside each fold (no new search), exactly like the reference models in
script 03, so the two tables are directly comparable.

Outputs: resultados/tpot_protocolos.csv   (same layout as modelos_protocolos.csv)
         resultados/tpot_resumo_buscas.csv (one row per finished search)
"""
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit

import config as C

warnings.filterwarnings("ignore")
TP = C.RES / "tpot"
DATA = {"original": C.DATA_ORIGINAL, "corrigido": C.DATA_CORRECTED}


def protocols(n):
    yield "random", list(ShuffleSplit(10, test_size=C.TEST_SIZE, random_state=C.SEED).split(np.arange(n)))
    yield "holdout", [C.holdout_index(n)]
    yield "blocked", list(C.blocked_folds(n))
    yield "walkfwd", list(C.walk_forward_folds(n))


def main():
    rows = []
    for tag, path in DATA.items():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        for sensor, target in C.PAIRS:
            pkl = TP / f"{tag}_seed{C.SEED}_{sensor}.pkl"
            if not pkl.exists():
                continue
            with open(pkl, "rb") as fh:
                pipe = pickle.load(fh)
            X, y = df[C.feature_cols(sensor)].to_numpy(float), df[target].to_numpy(float)
            for proto, folds in protocols(len(y)):
                m = pd.DataFrame([C.metrics(y[te], clone(pipe).fit(X[tr], y[tr]).predict(X[te]))
                                  for tr, te in folds])
                rows.append({"dataset": tag, "sensor": sensor, "target": target, "model": "TPOT-TS",
                             "protocol": proto, "n_folds": len(m),
                             **{f"{c}_mean": m[c].mean() for c in m},
                             **{f"{c}_sd": m[c].std(ddof=0) for c in m}})
            print(tag, sensor, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(C.RES / "tpot_protocolos.csv", index=False, float_format="%.4f")

    runs = []
    for fp in sorted(TP.glob("*.json")):
        d = json.loads(fp.read_text(encoding="utf-8"))
        runs.append({k: d.get(k) for k in ("dataset", "seed", "sensor", "R2_hold", "MAE_hold", "RMSE_hold",
                                           "R2_train", "overfit_gap", "n_generations", "n_evaluated",
                                           "duration_s", "pipeline")})
    runs = pd.DataFrame(runs)
    runs.to_csv(C.RES / "tpot_resumo_buscas.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 220)
    if len(out):
        print("\nTPOT-TS, MAE by protocol\n", out.pivot_table(index="sensor", columns=["dataset", "protocol"],
                                                              values="MAE_mean").round(2).to_string())
    if len(runs):
        print("\nSearch summary (holdout)\n", runs.drop(columns="pipeline").round(3).to_string(index=False))
        st = runs[runs.dataset == "corrigido"].groupby("sensor")[["R2_hold", "MAE_hold"]].agg(["mean", "min", "max", "count"])
        print("\nStability across seeds (corrected dataset)\n", st.round(3).to_string())


if __name__ == "__main__":
    main()
