# -*- coding: utf-8 -*-
"""Nested evaluation of the AutoML procedure under blocked cross-validation.

For every sensor-variable pair and every OUTER block (4 contiguous blocks, 1-h
embargo), a complete TPOT search is run on the outer training set only, with the
same configuration as the main search (internal 5-fold time-series split, 50
pipelines per generation, cap of 15 generations, early stop after 3, 20 min).
The selected pipeline is refitted on the outer training set and evaluated on the
outer test block, so that neither the architecture nor the parameters ever see
the block on which the procedure is scored.

Several workers may share the queue (.lock files); safe to interrupt and resume.
Outputs: resultados/tpot_aninhado/<sensor>_fold<k>.json (+ out-of-fold predictions)
Usage:  python 18_tpot_aninhado.py [--jobs 1,2,3,...] [--threads 12]
"""
import argparse
import asyncio
import importlib.util
import json
import sys
import time
import traceback
import warnings

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

import config as C

warnings.filterwarnings("ignore")
OUT = C.RES / "tpot_aninhado"
OUT.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location("tpot_main", C.ROOT / "scripts" / "04_tpot.py")
tpot_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tpot_main)


def search_and_score(X, y, tr, te, minutes, client):
    from sklearn.model_selection import TimeSeriesSplit
    from tpot import TPOTRegressor

    scaler = MinMaxScaler().fit(X[tr])
    est = TPOTRegressor(
        search_space=tpot_main.build_search_space(C.SEED), scorers=["r2"], scorers_weights=[1.0],
        cv=TimeSeriesSplit(n_splits=5), max_time_mins=minutes, max_eval_time_mins=max(2, minutes // 10),
        early_stop=3, generations=15, client=client, random_state=C.SEED, verbose=1)
    t0 = time.time()
    est.fit(scaler.transform(X[tr]), y[tr])
    dur = time.time() - t0
    best = est.fitted_pipeline_
    pred = make_pipeline(MinMaxScaler(), best).fit(X[tr], y[tr]).predict(X[te])
    hist, n_eval = tpot_main.history(est)
    return pred, {"pipeline": str(best), "n_generations": len(hist), "n_evaluated": n_eval, "duration_s": round(dur, 1),
                  **{k + "_test": v for k, v in C.metrics(y[te], pred).items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutos", type=int, default=20)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--jobs", type=str, default="")
    args = ap.parse_args()

    df = pd.read_csv(C.DATA_CORRECTED, index_col=0, parse_dates=True)
    folds = list(C.blocked_folds(len(df)))
    jobs = [(s, t, k) for s, t in C.PAIRS for k in range(len(folds))]
    order = [int(j) for j in args.jobs.split(",") if j.strip()] or list(range(1, len(jobs) + 1))

    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=1, threads_per_worker=args.threads, processes=False, dashboard_address=None)
    client = Client(cluster)
    print(f"[setup] {len(jobs)} nested searches | {args.minutos} min each | threads={args.threads}", flush=True)
    try:
        for i in order:
            sensor, target, k = jobs[i - 1]
            stem = OUT / f"{sensor}_fold{k}"
            if stem.with_suffix(".json").exists():
                continue
            try:
                open(stem.with_suffix(".lock"), "x").close()
            except FileExistsError:
                continue
            print(f"[{i}/{len(jobs)}] {stem.name}: searching...", flush=True)
            try:
                X, y = df[C.feature_cols(sensor)].to_numpy(float), df[target].to_numpy(float)
                tr, te = folds[k]
                pred, info = search_and_score(X, y, tr, te, args.minutos, client)
                info.update({"sensor": sensor, "target": target, "fold": k, "test_index": [int(te[0]), int(te[-1])],
                             "pred": [float(v) for v in pred]})
                stem.with_suffix(".json").write_text(json.dumps(info, indent=1), encoding="utf-8")
                print(f"[{i}/{len(jobs)}] {stem.name}: MAE_test={info['MAE_test']:.3f} ({info['duration_s']} s)", flush=True)
            except Exception:
                print(f"[{i}/{len(jobs)}] {stem.name}: ERROR\n{traceback.format_exc()}", flush=True)
                stem.with_suffix(".lock").unlink(missing_ok=True)
    finally:
        client.close()
        cluster.close()
    print("[done]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
