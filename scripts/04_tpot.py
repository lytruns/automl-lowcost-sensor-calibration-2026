# -*- coding: utf-8 -*-
"""TPOT-TS search on the revised datasets (same configuration as the submitted
manuscript: TPOT 1.1.0, 'linear' search space + PolynomialFeatures, internal
TimeSeriesSplit(5), early stop 3 generations, cap 15 generations).

The search sees ONLY the training portion (first 80 %). The selected pipeline
is wrapped with its MinMaxScaler and pickled, so that scripts 05/06 can refit
it under the other validation protocols and measure its inference footprint.

Run queue (checkpointed per file, safe to interrupt and resume):
  1. corrected dataset, seed 42      (primary result)
  2. original dataset,  seed 42      (like-for-like contrast)
  3. corrected dataset, seeds 7, 123 (search stability)

Several workers may share the queue: each run is claimed with a .lock file, and
--jobs sets the order in which a worker walks the queue (1-based indices).

Usage:  python 04_tpot.py [--minutos 20] [--threads 20] [--jobs 8,7,6,9,10]
"""
import argparse
import asyncio
import json
import pickle
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
OUT = C.RES / "tpot"
CV_FOLDS = 5
QUEUE = [("corrigido", 42), ("original", 42), ("corrigido", 7), ("corrigido", 123)]
DATA = {"corrigido": C.DATA_CORRECTED, "original": C.DATA_ORIGINAL}


def build_search_space(seed):
    from tpot.config import get_search_space
    from tpot.search_spaces.pipelines import (ChoicePipeline, DynamicUnionPipeline,
                                              SequentialPipeline, UnionPipeline)
    p = dict(random_state=seed)
    selectors = get_search_space(["selectors", "selectors_regression", "Passthrough"], **p)
    estimators = get_search_space(["regressors"], **p)
    scalers = get_search_space(["scalers", "Passthrough"], **p)
    transformers = UnionPipeline([
        ChoicePipeline([
            DynamicUnionPipeline(get_search_space(["transformers", "PolynomialFeatures"], **p)),
            get_search_space("SkipTransformer", **p),
        ]),
        get_search_space("Passthrough", **p),
    ])
    return SequentialPipeline(search_spaces=[scalers, selectors, transformers, estimators])


def history(est):
    try:
        ei = est.evaluated_individuals.copy()
        col = next(c for c in ei.columns if "r2" in str(c).lower())
        s = pd.to_numeric(ei[col], errors="coerce")
        ok = ei.assign(_s=s).dropna(subset=["_s"])
        ok = ok[ok["_s"] > -1e6]
        best, rows = float("-inf"), []
        for g, sub in ok.groupby("Generation"):
            best = max(best, float(sub["_s"].max()))
            rows.append({"gen": int(g), "best_so_far": round(best, 6),
                         "mean": round(float(sub["_s"].mean()), 6), "n": int(len(sub))})
        return rows, int(len(ei))
    except Exception as e:                                  # pragma: no cover
        return [{"error": f"{type(e).__name__}: {e}"}], None


def run(df, sensor, target, seed, minutes, client):
    from sklearn.model_selection import TimeSeriesSplit
    from tpot import TPOTRegressor

    X = df[C.feature_cols(sensor)].to_numpy(float)
    y = df[target].to_numpy(float)
    tr, va = C.holdout_index(len(y))
    scaler = MinMaxScaler().fit(X[tr])                      # fitted on training only

    est = TPOTRegressor(
        search_space=build_search_space(seed), scorers=["r2"], scorers_weights=[1.0],
        cv=TimeSeriesSplit(n_splits=CV_FOLDS), max_time_mins=minutes,
        max_eval_time_mins=max(2, minutes // 10), early_stop=3, generations=15,
        client=client, random_state=seed, verbose=1)
    t0 = time.time()
    est.fit(scaler.transform(X[tr]), y[tr])
    dur = time.time() - t0

    best = est.fitted_pipeline_
    hold = C.metrics(y[va], est.predict(scaler.transform(X[va])))
    train = C.metrics(y[tr], est.predict(scaler.transform(X[tr])))
    hist, n_eval = history(est)
    return {
        "sensor": sensor, "target": target, "seed": seed,
        **{f"{k}_hold": v for k, v in hold.items()},
        **{f"{k}_train": v for k, v in train.items()},
        "overfit_gap": train["R2"] - hold["R2"],
        "n_generations": len(hist), "n_evaluated": n_eval, "history": hist,
        "duration_s": round(dur, 1), "pipeline": str(best),
        "tpot_version": __import__("tpot").__version__,
        "sklearn_version": __import__("sklearn").__version__,
    }, make_pipeline(MinMaxScaler(), best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutos", type=int, default=20)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--jobs", type=str, default="", help="1-based queue indices, in the order to run")
    args = ap.parse_args()

    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=1, threads_per_worker=args.threads,
                           processes=False, dashboard_address=None)
    client = Client(cluster)
    jobs = [(tag, seed, s, t) for tag, seed in QUEUE for s, t in C.PAIRS]
    order = [int(j) for j in args.jobs.split(",") if j.strip()] or list(range(1, len(jobs) + 1))
    print(f"[setup] {len(jobs)} runs | {args.minutos} min/run | threads={args.threads}", flush=True)
    try:
        for i in order:
            tag, seed, sensor, target = jobs[i - 1]
            stem = OUT / f"{tag}_seed{seed}_{sensor}"
            if stem.with_suffix(".json").exists():
                print(f"[{i}/{len(jobs)}] {stem.name}: done, skipping", flush=True)
                continue
            try:                                            # claim the run (atomic create)
                open(stem.with_suffix(".lock"), "x").close()
            except FileExistsError:
                print(f"[{i}/{len(jobs)}] {stem.name}: claimed by another worker, skipping", flush=True)
                continue
            df = pd.read_csv(DATA[tag], index_col=0, parse_dates=True)
            print(f"[{i}/{len(jobs)}] {stem.name}: searching...", flush=True)
            try:
                res, pipe = run(df, sensor, target, seed, args.minutos, client)
                res["dataset"] = tag
                with open(stem.with_suffix(".pkl"), "wb") as fh:
                    pickle.dump(pipe, fh)
                stem.with_suffix(".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
                print(f"[{i}/{len(jobs)}] {stem.name}: R2_hold={res['R2_hold']:.3f} "
                      f"MAE={res['MAE_hold']:.3f} gap={res['overfit_gap']:+.3f} "
                      f"({res['duration_s']} s)", flush=True)
            except Exception:
                print(f"[{i}/{len(jobs)}] {stem.name}: ERROR\n{traceback.format_exc()}", flush=True)
    finally:
        client.close()
        cluster.close()
    print("[done]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
