# -*- coding: utf-8 -*-
"""Build the 5-min calibration datasets from the raw files.

Outputs (dados/processados/):
  dataset_original_5min.csv   timestamps used as recorded (as in the submitted
                              manuscript; low-cost clock 60 min ahead)
  dataset_corrigido_5min.csv  low-cost timestamps shifted by CLOCK_OFFSET_S

Both files share the same construction: raw low-cost readings (~3 s) are placed
on a 10-s grid and aggregated with a trailing 5-min window ending at each
reference stamp (Campbell end-of-interval convention). Rows with any missing
value are dropped. No other filtering or smoothing is applied.
"""
import json

import pandas as pd

import config as C

SENSORS = C.TEMP_SENSORS + C.RH_SENSORS


def load_station():
    st = pd.read_excel(C.STATION_FILE, sheet_name="Planilha1").dropna(how="all").iloc[:, :3]
    st.columns = ["timestamp", "temp", "umid"]
    st["timestamp"] = pd.to_datetime(st["timestamp"])
    return st.set_index("timestamp").astype(float).sort_index()


def load_device(path):
    x = pd.read_excel(path)
    x["ts"] = pd.to_datetime(dict(year=2000 + x.Ano, month=x.MesT, day=x.Dia,
                                  hour=x.Hora, minute=x.Minuto, second=x.Segundo),
                             errors="coerce")
    n_raw = len(x)
    x = x.dropna(subset=["ts"]).sort_values("ts").set_index("ts")[SENSORS]
    x = x.apply(pd.to_numeric, errors="coerce")
    grid = x.resample(f"{C.GRID_S}s").mean().interpolate(limit=6)   # gaps <= 1 min
    return grid, n_raw


def at_stamps(series, stamps, offset_s):
    """Trailing mean over [T + offset - window, T + offset) at each stamp T."""
    n = C.WINDOW_S // C.GRID_S
    r = series.rolling(n, min_periods=n // 2).mean()
    r.index = r.index - pd.Timedelta(seconds=offset_s)
    return r.reindex(stamps)


def main():
    st = load_station()
    devices, info = {}, {"station_rows": len(st),
                         "station_start": str(st.index.min()), "station_end": str(st.index.max())}
    for d, path in C.DEVICE_FILES.items():
        devices[d], n_raw = load_device(path)
        info[f"device_{d}_raw_rows"] = n_raw
        devices[d].to_pickle(C.PROC / f"_grid10s_device_{d}.pkl")    # cache for script 02

    for path, offset in [(C.DATA_ORIGINAL, 0), (C.DATA_CORRECTED, C.CLOCK_OFFSET_S)]:
        cols = {f"{d}_{s}": at_stamps(devices[d][s], st.index, offset)
                for d in C.REPLICAS for s in SENSORS}
        df = pd.DataFrame(cols).join(st)
        n_before = len(df)
        df = df.dropna()
        df.index.name = "timestamp"
        df.to_csv(path, float_format="%.4f")
        info[path.stem] = {"offset_s": offset, "rows": len(df), "dropped": n_before - len(df),
                           "start": str(df.index.min()), "end": str(df.index.max()),
                           "temp_range": [df.temp.min(), df.temp.max()],
                           "rh_range": [df.umid.min(), df.umid.max()]}
        print(path.name, info[path.stem])
    (C.PROC / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
