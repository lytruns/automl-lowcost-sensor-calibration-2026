# -*- coding: utf-8 -*-
"""Manuscript figures (clock-converted dataset only), written to MANUSCRITO_NOVO/figures as PDF + PNG.

  fig_workflow    methodological workflow
  fig_series      reference, raw and calibrated (quadratic, out-of-fold) series + scatter, BMP180 T and DHT22 RH
  fig_protocols   MAE by validation protocol and model
  fig_degree      in-sample vs out-of-sample MAE by polynomial degree
  fig_ablation    blocked-CV MAE vs number of units
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import config as C

OUT = C.ROOT / "MANUSCRITO_NOVO" / "figures"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e6e5e1"
COL = {"linear": "#2a78d6", "quadratic": "#eb6834", "RF": "#1baf7a", "TPOT-TS": "#eda100"}
LAB = {"linear": "Linear", "quadratic": "Quadratic", "RF": "Random forest", "TPOT-TS": "TPOT-TS"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("saved", name)


def results():
    r = pd.concat([pd.read_csv(C.RES / "modelos_protocolos.csv"), pd.read_csv(C.RES / "tpot_protocolos.csv")])
    return r[(r.dataset == "corrigido") & r.model.isin(COL)]


def fig_workflow():
    # Block diagram of the methodology. Text lines are kept short so that they fit
    # inside the boxes; the content mirrors Sections 2.3-2.7 of main.tex.
    import textwrap
    steps = [  # (heading, list of items; each item is wrapped to the box width)
        ("1. Acquisition", ["3 end devices, each with 5 sensor types (15 low-cost units); readings every ~3.6 s",
                            "Reference: Vaisala HMP45C + CR23X logger, 5-min records"]),
        ("2. Temporal harmonization", ["clocks converted to a common time base (UTC-3), verified by cross-correlation",
                                       "10-s grid; trailing 5-min means; 1,119 paired records"]),
        ("3. Calibration models", ["linear (4 coef.); quadratic (10 coef.)", "random forest",
                                   "TPOT AutoML, pipeline search nested in each training set"]),
        ("4. Validation protocols", ["random split (leakage check)", "temporal holdout",
                                     "blocked CV, 1-h embargo (primary)", "walk-forward"]),
        ("5. Statistical assessment", ["paired moving-block bootstrap", "Diebold-Mariano test + Holm correction",
                                       "equivalence within practical-relevance margins", "seed stability; polynomial degree"]),
        ("6. Deployment analyses", ["replica ablation (1 to 3 units)", "inference footprint",
                                    "dynamic response and heating-cooling asymmetry", "node vs. gateway"]),
    ]
    # Two rows of three boxes, read left to right and then down (snake connector between the rows).
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_axis_off(); ax.set_xlim(0, 3); ax.set_ylim(0, 2)
    W, H, GAP = 0.92, 0.82, 0.08
    for i, (head, items) in enumerate(steps):
        col, row = i % 3, i // 3
        x0, y0 = col + (1 - W) / 2, (1.10 if row == 0 else 0.08)
        body = "\n".join(textwrap.fill(it, width=34) for it in items)
        ax.add_patch(FancyBboxPatch((x0, y0), W, H, boxstyle="round,pad=0.01,rounding_size=0.04", fc="#f4f3f0", ec=MUTED, lw=0.8))
        ax.text(x0 + W / 2, y0 + H - 0.12, head, ha="center", va="center", fontsize=7.8, fontweight="bold", color=INK)
        ax.text(x0 + W / 2, y0 + (H - 0.24) / 2, body, ha="center", va="center", fontsize=6.2, color=INK, linespacing=1.3)
        if col < 2:  # arrow to the next box in the same row
            yc = y0 + H / 2
            ax.annotate("", xy=(col + 1 + (1 - W) / 2, yc), xytext=(x0 + W, yc), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))
    # connector from box 3 (end of row 1) to box 4 (start of row 2)
    xa, ya = 2.5, 1.10            # bottom centre of box 3
    xb, yb = 0.5, 0.10 + H        # top centre of box 4
    ym = 1.10 - GAP / 2 - 0.02    # horizontal run between the rows
    ax.plot([xa, xa, xb], [ya, ym, ym], color=INK, lw=0.9, solid_capstyle="round")
    ax.annotate("", xy=(xb, yb), xytext=(xb, ym), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))
    save(fig, "fig_workflow")


def fig_series():
    oof = pd.read_csv(C.RES / "predicoes_oof.csv", parse_dates=["timestamp"]).set_index("timestamp")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), gridspec_kw={"width_ratios": [2.6, 1]})
    for row, (s, unit, name) in enumerate([("BMP180_T", "Air temperature (°C)", "BMP180"), ("DHT22_U", "Relative humidity (% RH)", "DHT22")]):
        a = axes[row, 0]
        a.plot(oof.index, oof[f"{s}__raw"], color=MUTED, lw=1.0, label="Low-cost, uncalibrated")
        a.plot(oof.index, oof[f"{s}__ref"], color=INK, lw=1.6, label="Reference (HMP45C)")
        a.plot(oof.index, oof[f"{s}__quadratic"], color=COL["quadratic"], lw=1.2, label="Low-cost, quadratic calibration")
        a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        a.xaxis.set_major_locator(mdates.DayLocator())
        a.set_ylabel(unit); a.set_title(f"{name}", loc="left", fontsize=9)
        if row == 0:
            a.legend(loc="upper center", ncol=3, fontsize=7.2, bbox_to_anchor=(0.5, 1.32))
        b = axes[row, 1]
        y, p = oof[f"{s}__ref"], oof[f"{s}__quadratic"]
        b.scatter(y, p, s=4, color=COL["quadratic"], alpha=0.35, linewidths=0)
        lim = [min(y.min(), p.min()), max(y.max(), p.max())]
        b.plot(lim, lim, color=INK, lw=0.8)
        b.set_xlabel("Reference"); b.set_ylabel("Calibrated")
        b.text(0.04, 0.93, f"MAE = {np.abs(y - p).mean():.2f}", transform=b.transAxes, fontsize=7.5, va="top")
    fig.tight_layout()
    save(fig, "fig_series")


def fig_protocols():
    r = results()
    protos = {"random": "Random\nsplit", "holdout": "Temporal\nholdout", "blocked": "Blocked\nCV", "walkfwd": "Walk-\nforward"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for a, (t, unit, title) in zip(axes, [("temp", "°C", "Air temperature"), ("umid", "% RH", "Relative humidity")]):
        g = r[r.target == t].groupby(["protocol", "model"]).MAE_mean.mean().unstack().reindex(list(protos))[list(COL)]
        w = 0.8 / 4
        for k, m in enumerate(g.columns):
            x = np.arange(4) + (k - 1.5) * w
            a.bar(x, g[m], width=w * 0.86, color=COL[m], label=LAB[m], linewidth=0)
            for xi, v in zip(x, g[m]):
                if not np.isfinite(v):          # TPOT-TS is shown only for holdout and (nested) blocked CV
                    continue
                a.annotate(f"{v:.2f}" if t == "temp" else f"{v:.1f}", (xi, v), xytext=(0, 2.5), textcoords="offset points",
                           ha="center", va="bottom", fontsize=6, rotation=90, color=INK)
        a.set_xticks(np.arange(4), list(protos.values())); a.grid(axis="x", visible=False)
        a.set_ylim(0, g.max().max() * 1.28); a.set_ylabel(f"MAE ({unit}), mean over sensors"); a.set_title(title, loc="left", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07), fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_protocols")


def fig_degree():
    g = pd.read_csv(C.RES / "grau_polinomio.csv"); g = g[g.inputs == "3 replicas"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    series = [("MAE_insample", "In-sample", MUTED), ("MAE_blocked", "Blocked CV", COL["quadratic"]), ("MAE_walkfwd", "Walk-forward", COL["linear"])]
    for a, (t, unit, title) in zip(axes, [("temp", "°C", "Air temperature"), ("umid", "% RH", "Relative humidity")]):
        m = g[g.target == t].groupby("degree")[[s[0] for s in series]].mean()
        for col, lab, c in series:
            a.plot(m.index, m[col], color=c, lw=2, marker="o", ms=5, mec="white", mew=1.1, label=lab)
        a.set_yscale("log"); a.set_xticks(range(1, 6)); a.set_xlabel("Polynomial degree"); a.set_ylabel(f"MAE ({unit}), log scale")
        a.set_title(title, loc="left", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_degree")


def fig_ablation():
    ab = pd.read_csv(C.RES / "ablacao_replicas.csv"); ab = ab[ab.model.isin(["linear", "quadratic", "RF"])].copy()
    ab["target"] = np.where(ab.sensor.str.endswith("_T"), "temp", "umid")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for a, (t, unit, title) in zip(axes, [("temp", "°C", "Air temperature"), ("umid", "% RH", "Relative humidity")]):
        m = ab[ab.target == t].groupby(["replicas", "model"]).MAE_mean.mean().unstack()
        for k in ("linear", "quadratic", "RF"):
            a.plot(m.index, m[k], color=COL[k], lw=2, marker="o", ms=5.5, mec="white", mew=1.2, label=LAB[k])
            a.annotate(f"{m[k].iloc[-1]:.2f}", (3, m[k].iloc[-1]), xytext=(7, 0), textcoords="offset points", va="center", fontsize=7.5)
        a.set_xticks([1, 2, 3]); a.set_xlim(0.85, 3.45)
        a.set_xlabel("Units of the same sensor type used as inputs"); a.set_ylabel(f"Blocked-CV MAE ({unit})")
        a.set_title(title, loc="left", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_ablation")


if __name__ == "__main__":
    fig_workflow(); fig_series(); fig_protocols(); fig_degree(); fig_ablation()
