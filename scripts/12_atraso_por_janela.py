# -*- coding: utf-8 -*-
"""Is the ~1-h delay physical (sensor response) or a constant clock offset?

A physical response lag depends on the sensor type, on ventilation/radiation
(day vs night) and on how fast conditions change.  A clock offset is the same
constant everywhere.  Using the timestamps AS RECORDED (no correction), the
delay is estimated independently in consecutive 8-h windows, for each sensor
type and variable (10-s resolution, 0-120 min search).

Outputs: resultados/atraso_por_janela.csv, figuras/fig5_eventos_relogio.png
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

warnings.filterwarnings("ignore")
SHIFTS = np.arange(0, 7201, 30)


def at_stamps(series, stamps, offset_s, win):
    n = max(1, win // C.GRID_S)
    r = series.rolling(n, min_periods=max(1, n // 2)).mean()
    r.index = r.index - pd.Timedelta(seconds=int(offset_s))
    return r.reindex(stamps)


def main():
    ref = pd.read_csv(C.DATA_ORIGINAL, index_col=0, parse_dates=True)[["temp", "umid"]]
    dev = {d: pd.read_pickle(C.PROC / f"_grid10s_device_{d}.pkl") for d in C.REPLICAS}
    low = {s: sum(dev[d][s] for d in C.REPLICAS) / 3 for s, _ in C.PAIRS}
    starts = pd.date_range("2018-11-11 12:00", "2018-11-15 02:00", freq="8h")
    rows = []
    for s, target in C.PAIRS:
        win = 300 if target == "temp" else 60
        shifted = {k: at_stamps(low[s], ref.index, k, win).diff() for k in SHIFTS}
        dref = ref[target].diff()
        for t0 in starts:
            sl = slice(t0, t0 + pd.Timedelta("8h"))
            sc = []
            for k in SHIFTS:
                a, b = shifted[k][sl], dref[sl]; ok = a.notna() & b.notna()
                sc.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 30 else np.nan)
            sc = np.array(sc)
            rows.append({"sensor": s, "window_start": t0, "period": "day" if 6 <= (t0.hour + 4) % 24 < 18 else "night",
                         "delay_min": SHIFTS[int(np.nanargmax(sc))] / 60, "r_peak": np.nanmax(sc)})
    out = pd.DataFrame(rows)
    out.to_csv(C.RES / "atraso_por_janela.csv", index=False, float_format="%.3f")
    good = out[out.r_peak >= 0.5]
    pd.set_option("display.width", 220)
    print("Delay (min) per 8-h window, timestamps as recorded; only windows with a clear peak (r >= 0.5)\n")
    t = good.pivot(index="window_start", columns="sensor", values="delay_min")
    t.index = t.index.strftime("%d/%m %Hh")
    print(t.round(1).to_string())
    print("\nSummary by sensor:\n", good.groupby("sensor").delay_min.agg(["count", "median", "min", "max"]).round(1).to_string())
    print("\nSummary by period:\n", good.groupby("period").delay_min.agg(["count", "median", "min", "max"]).round(1).to_string())
    print(f"\nAll windows/sensors: median {good.delay_min.median():.1f} min, IQR "
          f"{good.delay_min.quantile(.25):.1f}-{good.delay_min.quantile(.75):.1f}, n = {len(good)}")

    # figure: three abrupt events, series standardised, clock as recorded
    orig = pd.read_csv(C.DATA_ORIGINAL, index_col=0, parse_dates=True)
    events = [("2018-11-13 12:30", "2018-11-13 17:30", "BMP180_T", "temp", "Afternoon drop, 13 Nov (temperature)"),
              ("2018-11-14 04:30", "2018-11-14 09:30", "DHT22_T", "temp", "Sunrise, 14 Nov (temperature)"),
              ("2018-11-13 12:30", "2018-11-13 17:30", "DHT22_U", "umid", "Afternoon rise, 13 Nov (humidity)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    z = lambda x: (x - x.mean()) / x.std()
    for a, (t0, t1, s, target, title) in zip(axes, events):
        w = orig.loc[t0:t1]
        lowm = w[C.feature_cols(s)].mean(axis=1)
        a.plot(w.index, z(w[target]), color="#0b0b0b", lw=2, label="Reference (its own clock)")
        a.plot(w.index, z(lowm), color="#eb6834", lw=2, label=f"{s.split('_')[0]} (its own clock)")
        a.plot(w.index - pd.Timedelta("60min"), z(lowm), color="#1baf7a", lw=1.6, ls=(0, (4, 2)), label="same, moved 60 min")
        a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        a.set_xlim(w.index[0], w.index[-1]); a.set_title(title, loc="left", fontsize=9)
        a.set_ylabel("Standardised value"); a.grid(color="#e6e5e1", lw=0.6)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    h, l = axes[0].get_legend_handles_labels(); l[1] = "Low-cost sensor (its own clock)"
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08), fontsize=9)
    fig.tight_layout(); fig.savefig(C.FIG / "fig5_eventos_relogio.png", dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
