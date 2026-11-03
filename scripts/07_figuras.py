# -*- coding: utf-8 -*-
"""Figures of the revised analysis (PNG 300 dpi + PDF), built only from the CSVs
written by scripts 02-06.

  fig1_relogio       clock offset: cross-correlation curve and a morning transition
  fig2_vazamento     MAE by validation protocol and model, original vs corrected
  fig3_antes_depois  blocked-CV MAE per sensor, original vs corrected dataset
  fig4_ablacao       blocked-CV MAE versus number of replicas

Categorical colours follow a fixed model order (validated for colour-vision
deficiency); every figure uses a single y-scale per panel.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e6e5e1"
COLORS = {"linear": "#2a78d6", "quadratic": "#eb6834", "RF": "#1baf7a", "TPOT-TS": "#eda100"}
LABEL = {"linear": "Linear", "quadratic": "Quadratic", "RF": "Random forest", "TPOT-TS": "TPOT-TS"}
PROTO = {"random": "Random\nsplit", "holdout": "Temporal\nholdout", "blocked": "Blocked\nCV", "walkfwd": "Walk-\nforward"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 120, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(C.FIG / f"{name}.png", dpi=300)
    fig.savefig(C.FIG / f"{name}.pdf")
    plt.close(fig)
    print("saved", name)


def all_models():
    df = pd.read_csv(C.RES / "modelos_protocolos.csv")
    tp = C.RES / "tpot_protocolos.csv"
    if tp.exists() and tp.stat().st_size > 10:
        df = pd.concat([df, pd.read_csv(tp)], ignore_index=True)
    return df


def fig_clock():
    xc = pd.read_csv(C.RES / "relogio_xcorr.csv", index_col=0)
    orig = pd.read_csv(C.DATA_ORIGINAL, index_col=0, parse_dates=True)
    corr = pd.read_csv(C.DATA_CORRECTED, index_col=0, parse_dates=True)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})

    a = ax[0]
    for c in xc.columns:
        a.plot(xc.index, xc[c], color=MUTED, lw=0.5, alpha=0.45)
    a.plot(xc.index, xc.median(axis=1), color=COLORS["linear"], lw=2, label="Median of 24 channels")
    a.axvline(60, color=INK, lw=0.8, ls=(0, (4, 3)))
    a.annotate("60 min\n(DST shift)", (60, xc.median(axis=1).max()), xytext=(66, 0.62), fontsize=8, color=INK, va="top")
    a.set(xlabel="Shift applied to the low-cost timestamps (min)", ylabel="Correlation of first differences\nwith the reference",
          xlim=(0, 90), ylim=(-0.1, 0.85))
    a.legend(loc="upper left", fontsize=8)
    a.set_title("(a) Cross-correlation, training portion", loc="left", fontsize=9)

    b = ax[1]
    w = slice("2018-11-14 04:00", "2018-11-14 10:00")
    cols = C.feature_cols("BMP180_T")
    b.plot(orig.loc[w].index, orig.loc[w, "temp"], color=INK, lw=2, label="Reference (HMP45C)")
    b.plot(orig.loc[w].index, orig.loc[w, cols].mean(axis=1), color=COLORS["quadratic"], lw=2, label="BMP180, as recorded")
    b.plot(corr.loc[w].index, corr.loc[w, cols].mean(axis=1), color=COLORS["RF"], lw=2, label="BMP180, clock corrected")
    b.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    b.set(xlabel="Reference clock, 14 Nov 2018 (UTC-3)", ylabel="Air temperature (°C)")
    b.legend(loc="upper left", fontsize=8)
    b.set_title("(b) Morning transition, uncalibrated readings", loc="left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig1_relogio")


def fig_leakage():
    df = all_models()
    df = df[df.model.isin(COLORS)]
    models = [m for m in COLORS if m in set(df.model)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharey="row")
    for i, (target, unit) in enumerate([("temp", "°C"), ("umid", "% RH")]):
        for j, tag in enumerate(["original", "corrigido"]):
            a = axes[i, j]
            sub = df[(df.target == target) & (df.dataset == tag)].groupby(["protocol", "model"]).MAE_mean.mean().unstack()
            sub = sub.reindex(list(PROTO)).reindex(columns=[m for m in models if m in sub.columns])
            w = 0.8 / len(sub.columns)
            for k, m in enumerate(sub.columns):
                x = np.arange(len(sub)) + (k - (len(sub.columns) - 1) / 2) * w
                a.bar(x, sub[m], width=w * 0.86, color=COLORS[m], label=LABEL[m], linewidth=0)
                for xi, v in zip(x, sub[m]):
                    if np.isfinite(v):
                        a.annotate(f"{v:.2f}" if target == "temp" else f"{v:.1f}", (xi, v), xytext=(0, 2.5),
                                   textcoords="offset points", ha="center", va="bottom", fontsize=6.2,
                                   color=INK, rotation=90)
            a.set_xticks(np.arange(len(sub)), [PROTO[p] for p in sub.index])
            a.grid(axis="x", visible=False)
            a.set_ylim(0, np.nanmax(df[df.target == target].groupby(["dataset", "protocol", "model"]).MAE_mean.mean()) * 1.25)
            if j == 0:
                a.set_ylabel(f"MAE ({unit})\nmean over sensors")
            a.set_title(("Temperature" if target == "temp" else "Relative humidity") +
                        (" | timestamps as recorded" if tag == "original" else " | clock corrected"), loc="left", fontsize=9)
    h, l = max((a.get_legend_handles_labels() for a in axes.ravel()), key=lambda hl: len(hl[1]))
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.03), fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig2_vazamento")


def fig_before_after():
    df = all_models()
    df = df[(df.protocol == "blocked") & df.model.isin(COLORS)]
    best = df.loc[df.groupby(["dataset", "sensor"]).MAE_mean.idxmin()]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [5, 3]})
    for a, (target, unit, sensors) in zip(axes, [("temp", "°C", C.TEMP_SENSORS), ("umid", "% RH", C.RH_SENSORS)]):
        for k, s in enumerate(sensors):
            o = best[(best.sensor == s) & (best.dataset == "original")].MAE_mean.item()
            c = best[(best.sensor == s) & (best.dataset == "corrigido")].MAE_mean.item()
            a.plot([o, c], [k, k], color=MUTED, lw=1.2, zorder=1)
            a.scatter([o], [k], s=42, color=COLORS["quadratic"], zorder=2, label="As recorded" if k == 0 else None)
            a.scatter([c], [k], s=42, color=COLORS["RF"], zorder=2, label="Clock corrected" if k == 0 else None)
            a.text(o, k + 0.22, f"{o:.2f}", ha="center", fontsize=7.5)
            a.text(c, k + 0.22, f"{c:.2f}", ha="center", fontsize=7.5)
        a.set_yticks(range(len(sensors)), [s.split("_")[0] for s in sensors])
        a.set_ylim(-0.6, len(sensors) - 0.3)
        a.set_xlim(0, best[best.sensor.isin(sensors)].MAE_mean.max() * 1.12)
        a.grid(axis="y", visible=False)
        a.set_xlabel(f"Blocked-CV MAE, best model ({unit})")
        a.set_title("Temperature" if target == "temp" else "Relative humidity", loc="left", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07), fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig3_antes_depois")


def fig_ablation():
    ab = pd.read_csv(C.RES / "ablacao_replicas.csv")
    ab["target"] = np.where(ab.sensor.str.endswith("_T"), "temp", "umid")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for a, (target, unit) in zip(axes, [("temp", "°C"), ("umid", "% RH")]):
        sub = ab[ab.target == target].groupby(["model", "replicas"]).MAE_mean.mean().unstack(0)
        for m in [m for m in COLORS if m in sub.columns]:
            a.plot(sub.index, sub[m], color=COLORS[m], lw=2, marker="o", ms=5.5, mec="white", mew=1.2, label=LABEL[m])
            a.annotate(f"{sub[m].iloc[-1]:.2f}", (3, sub[m].iloc[-1]), xytext=(7, 0), textcoords="offset points",
                       va="center", fontsize=7.5)
        a.set_xticks([1, 2, 3])
        a.set_xlim(0.85, 3.45)
        a.set(xlabel="Units of the same sensor used as features", ylabel=f"Blocked-CV MAE ({unit})")
        a.set_title("Temperature" if target == "temp" else "Relative humidity", loc="left", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.07), fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig4_ablacao")


if __name__ == "__main__":
    fig_clock()
    fig_leakage()
    fig_before_after()
    fig_ablation()
