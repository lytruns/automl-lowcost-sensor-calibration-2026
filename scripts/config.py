# -*- coding: utf-8 -*-
"""Shared configuration for the revised analysis (REVISAO_20_09_2026).

All paths are relative to the revision folder, so the pipeline can be moved or
published as-is.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dados" / "brutos"
PROC = ROOT / "dados" / "processados"
RES = ROOT / "resultados"
FIG = RES / "figuras"
for _p in (PROC, RES, FIG, RES / "tpot"):
    _p.mkdir(parents=True, exist_ok=True)

STATION_FILE = RAW / "Dados da estação estufa.xlsx"
DEVICE_FILES = {d: RAW / f"Dispositivo {d}.xlsx" for d in "ABC"}
DATA_ORIGINAL = PROC / "dataset_original_5min.csv"
DATA_CORRECTED = PROC / "dataset_corrigido_5min.csv"

# --- Clock correction -------------------------------------------------------
# The RTC of the low-cost system was set to Brazilian civil time after the
# start of daylight saving time (DST, 00:00 of 4 Nov 2018, UTC-2), whereas the
# CR23X reference logger remained on standard time (UTC-3). The low-cost
# timestamps are therefore 60 min ahead of the reference. The correction is
# fixed a priori from this physical cause (NOT tuned on the data); script 02
# verifies it empirically using the training portion only.
CLOCK_OFFSET_S = 3600

# Reference logger convention (Campbell): each 5-min record is stamped at the
# END of its averaging interval. The low-cost readings are aggregated with the
# same convention (trailing window).
WINDOW_S = 300
GRID_S = 10

TEMP_SENSORS = ["BMP180_T", "BME280_T", "DHT22_T", "LM35_T", "DHT11_T"]
RH_SENSORS = ["BME280_U", "DHT22_U", "DHT11_U"]
PAIRS = [(s, "temp") for s in TEMP_SENSORS] + [(s, "umid") for s in RH_SENSORS]
REPLICAS = "ABC"

# --- Validation protocols ---------------------------------------------------
TEST_SIZE = 0.20          # temporal holdout: final 20 %
N_BLOCKS = 4              # blocked CV: ~1 day per contiguous block
EMBARGO = 12              # samples (1 h) dropped on each side of a test block
WF_FOLDS = 5              # walk-forward (expanding window)
SEED = 42

# Site (Botucatu, SP) for the sunrise check
LAT, LON = -22.85, -48.43


def feature_cols(sensor, replicas=REPLICAS):
    return [f"{r}_{sensor}" for r in replicas]


def metrics(y, p):
    return {
        "R2": float(r2_score(y, p)),
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(mean_squared_error(y, p) ** 0.5),
    }


def holdout_index(n):
    k = int(round(n * (1 - TEST_SIZE)))
    return np.arange(k), np.arange(k, n)


def blocked_folds(n, k=N_BLOCKS, embargo=EMBARGO):
    """Contiguous test blocks; training excludes an embargo around each block."""
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        te = np.arange(edges[i], edges[i + 1])
        tr = np.r_[0:max(0, edges[i] - embargo), min(n, edges[i + 1] + embargo):n]
        yield tr, te


def walk_forward_folds(n, k=WF_FOLDS, embargo=EMBARGO):
    """Expanding-window (rolling-origin) evaluation: train on the past only."""
    edges = np.linspace(0, n, k + 2).astype(int)
    for i in range(1, k + 1):
        yield np.arange(0, max(1, edges[i] - embargo)), np.arange(edges[i], edges[i + 1])
