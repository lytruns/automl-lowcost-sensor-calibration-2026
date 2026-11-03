# -*- coding: utf-8 -*-
"""Empirical verification of the 60-min clock offset (DST) between the low-cost
system and the CR23X reference. Three independent lines of evidence:

  (a) cross-correlation of FIRST DIFFERENCES between each low-cost channel and
      the reference, as a function of the time shift (30-s resolution), using
      the training portion only (first 80 %);
  (b) morning warming onset on each clock versus the computed sunrise;
  (c) sensitivity of a simple calibration model (quadratic, blocked CV inside
      the training portion) to the assumed offset.

Outputs: resultados/relogio_xcorr.csv, relogio_offsets.csv,
         relogio_nascer_do_sol.csv, relogio_sensibilidade.csv
"""
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

import config as C

warnings.filterwarnings("ignore")
SENSORS = C.TEMP_SENSORS + C.RH_SENSORS
OFFSETS = np.arange(0, 5401, 30)


def at_stamps(series, stamps, offset_s):
    n = C.WINDOW_S // C.GRID_S
    r = series.rolling(n, min_periods=n // 2).mean()
    r.index = r.index - pd.Timedelta(seconds=int(offset_s))
    return r.reindex(stamps)


def diff_corr(a, b):
    da, db = a.diff(), b.diff()
    ok = da.notna() & db.notna()
    return float(np.corrcoef(da[ok], db[ok])[0, 1])


def sunrise_minutes_utc3(date):
    """NOAA approximation; minutes after midnight, Brasília standard time."""
    n = date.timetuple().tm_yday
    g = 2 * math.pi / 365 * (n - 1 + 0.5)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    dec = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
           - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
           - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    ha = math.degrees(math.acos(math.cos(math.radians(90.833))
                                / (math.cos(math.radians(C.LAT)) * math.cos(dec))
                                - math.tan(math.radians(C.LAT)) * math.tan(dec)))
    return 720 - 4 * (C.LON + ha) - eqt - 180


def hhmm(m):
    return f"{int(m // 60):02d}:{int(m % 60):02d}"


def onset(series, day, rise=0.5):
    x = series[f"{day} 03:00":f"{day} 09:30"]
    tmin = x[:f"{day} 07:00"].idxmin()
    up = x[(x.index > tmin) & (x > x[tmin] + rise)]
    return up.index[0].strftime("%H:%M")


def main():
    st = pd.read_csv(C.DATA_ORIGINAL, index_col=0, parse_dates=True)[["temp", "umid"]]
    dev = {d: pd.read_pickle(C.PROC / f"_grid10s_device_{d}.pkl") for d in C.REPLICAS}
    tr = st.index[C.holdout_index(len(st))[0]]

    # (a) cross-correlation of first differences, training portion only
    curves, best = {}, []
    for s in SENSORS:
        ref = st["temp" if s.endswith("_T") else "umid"].loc[tr]
        for d in C.REPLICAS:
            c = [diff_corr(at_stamps(dev[d][s], tr, o), ref) for o in OFFSETS]
            curves[f"{d}_{s}"] = c
            k = int(np.argmax(c))
            best.append({"device": d, "sensor": s, "best_offset_min": OFFSETS[k] / 60,
                         "r_best": c[k], "r_at_0": c[0], "r_at_60": c[list(OFFSETS).index(3600)]})
    xc = pd.DataFrame(curves, index=OFFSETS / 60)
    xc.index.name = "offset_min"
    xc.to_csv(C.RES / "relogio_xcorr.csv", float_format="%.4f")
    best = pd.DataFrame(best)
    best.to_csv(C.RES / "relogio_offsets.csv", index=False, float_format="%.3f")
    print("(a) best offset (min), by channel — training portion only")
    print(best.pivot(index="sensor", columns="device", values="best_offset_min").to_string())
    print(f"    overall: median {best.best_offset_min.median():.1f} min | "
          f"IQR {best.best_offset_min.quantile(.25):.1f}-{best.best_offset_min.quantile(.75):.1f} | "
          f"mean r at 0 min = {best.r_at_0.mean():.3f}, at 60 min = {best.r_at_60.mean():.3f}, "
          f"at best = {best.r_best.mean():.3f}")

    # (b) sunrise versus warming onset on each clock
    orig = pd.read_csv(C.DATA_ORIGINAL, index_col=0, parse_dates=True)
    low = orig[[c for c in orig.columns if c.endswith("_T")]].mean(axis=1)
    rows = []
    for day in ["2018-11-12", "2018-11-13", "2018-11-14", "2018-11-15"]:
        sr = sunrise_minutes_utc3(pd.Timestamp(day))
        rows.append({"day": day, "sunrise_standard_UTC-3": hhmm(sr), "sunrise_DST_UTC-2": hhmm(sr + 60),
                     "onset_reference_clock": onset(orig["temp"], day),
                     "onset_lowcost_clock": onset(low, day)})
    sun = pd.DataFrame(rows)
    sun.to_csv(C.RES / "relogio_nascer_do_sol.csv", index=False)
    print("\n(b) sunrise vs. warming onset\n", sun.to_string(index=False))

    # (c) sensitivity of a quadratic calibration to the assumed offset (training portion)
    rows = []
    for off_min in range(0, 91, 5):
        for s, target in C.PAIRS:
            X = pd.concat([at_stamps(dev[d][s], tr, off_min * 60) for d in C.REPLICAS], axis=1)
            ok = X.notna().all(axis=1)
            Xv, y = X[ok].to_numpy(), st[target].loc[tr][ok].to_numpy()
            err = []
            for a, b in C.blocked_folds(len(y), k=3):
                m = make_pipeline(MinMaxScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression()).fit(Xv[a], y[a])
                err.append(np.abs(y[b] - m.predict(Xv[b])).mean())
            rows.append({"offset_min": off_min, "sensor": s, "MAE": float(np.mean(err))})
    sens = pd.DataFrame(rows)
    sens.to_csv(C.RES / "relogio_sensibilidade.csv", index=False, float_format="%.4f")
    print("\n(c) quadratic blocked-CV MAE inside the training portion, by assumed offset (min)")
    print(sens.pivot(index="offset_min", columns="sensor", values="MAE").round(2).to_string())


if __name__ == "__main__":
    main()
