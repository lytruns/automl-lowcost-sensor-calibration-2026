# -*- coding: utf-8 -*-
"""Supplementary material: one figure and the tables, generated from the result CSVs.

  fig_S1_alignment     correlation of first differences vs residual time shift (common time base)
  tab_S1_alignment     residual shift per sensor type (mean +/- s.d. of three units)
  tab_S2_pipelines     pipelines selected by TPOT (three seeds)
  tab_S3_blocklength   number of significant paired differences by bootstrap block length and DM-Holm
  tab_S4_dm            Diebold-Mariano statistics per sensor and model pair
  tab_S5_allmetrics    MAE / RMSE / R2 for every sensor, model and protocol
  (tab_rh_support is produced by script 14)
"""
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

warnings.filterwarnings("ignore")
M = C.ROOT / "MANUSCRITO_NOVO"
TAB, FIGD = M / "tables", M / "figures"
NAME = {s: s.split("_")[0] for s, _ in C.PAIRS}
HEAD = {"linear": "Linear", "quadratic": "Quadratic", "RF": "RF", "HGB": "HGB", "TPOT-TS": "TPOT-TS", "raw": "Raw"}
NL = r" \\"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e6e5e1"


def math_signs(ln):
    ln = re.sub(r"(?<![\d\-$])([+-]\d+(?:\.\d+)?)", r"$\1$", ln)
    return re.sub(r"\$[+-](0\.0+)\$", r"$\1$", ln)


def write(name, colspec, header, body, longtable=False):
    body = [math_signs(b) for b in body]
    if longtable:
        text = "\n".join([rf"\begin{{longtable}}{{{colspec}}}", r"\toprule", *header, r"\midrule", r"\endhead",
                          r"\bottomrule", r"\endfoot", *body, r"\end{longtable}", ""])
    else:
        text = "\n".join([rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *header, r"\midrule", *body,
                          r"\bottomrule", r"\end{tabular}", ""])
    (TAB / f"{name}.tex").write_text(text, encoding="utf-8")
    print("wrote", name)


def at_stamps(series, stamps, offset_s, win_s):
    n = max(1, win_s // C.GRID_S)
    r = series.rolling(n, min_periods=max(1, n // 2)).mean()
    r.index = r.index - pd.Timedelta(seconds=int(offset_s))
    return r.reindex(stamps)


def alignment():
    df = pd.read_csv(C.DATA_CORRECTED, index_col=0, parse_dates=True)
    dev = {d: pd.read_pickle(C.PROC / f"_grid10s_device_{d}.pkl") for d in C.REPLICAS}
    shifts = np.arange(-1800, 1801, 30)
    curves, rows = {}, []
    for s, target in C.PAIRS:
        win = 300 if target == "temp" else 60
        dref = df[target].diff()
        best = []
        for d in C.REPLICAS:
            sc = []
            for k in shifts:
                a = at_stamps(dev[d][s], df.index, C.CLOCK_OFFSET_S + k, win).diff()
                ok = a.notna() & dref.notna()
                sc.append(np.corrcoef(a[ok], dref[ok])[0, 1])
            curves[f"{d}_{s}"] = sc
            best.append((shifts[int(np.argmax(sc))] / 60, max(sc), sc[list(shifts).index(0)]))
        b = np.array(best)
        rows.append(NAME[s] + " & " + ("Air temperature" if target == "temp" else "Relative humidity")
                    + " & {:+.1f} $\\pm$ {:.1f} & {:.2f} & {:.2f}".format(b[:, 0].mean(), b[:, 0].std(), b[:, 1].mean(), b[:, 2].mean()) + NL)
    write("tab_S1_alignment", "@{}ll ccc@{}",
          [r"Sensor & Variable & Residual shift (min) & $r$ at best shift & $r$ at zero shift" + NL], rows)

    xc = pd.DataFrame(curves, index=shifts / 60)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for a, (suffix, title) in zip(axes, [("_T", "Air temperature (15 channels)"), ("_U", "Relative humidity (9 channels)")]):
        cols = [c for c in xc.columns if c.endswith(suffix)]
        for c in cols:
            a.plot(xc.index, xc[c], color=MUTED, lw=0.5, alpha=0.5)
        a.plot(xc.index, xc[cols].median(axis=1), color="#2a78d6", lw=2, label="Median")
        a.axvline(0, color=INK, lw=0.8, ls=(0, (4, 3)))
        a.set_xlabel("Additional shift applied to the low-cost series (min)"); a.set_title(title, loc="left", fontsize=9)
    axes[0].set_ylabel("Correlation of first differences\nwith the reference"); axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGD / "fig_S1_alignment.pdf", bbox_inches="tight"); fig.savefig(FIGD / "fig_S1_alignment.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("saved fig_S1_alignment")


def pipelines():
    tp = pd.read_csv(C.RES / "tpot_resumo_buscas.csv")
    tp = tp[tp.dataset == "corrigido"]
    skip = {"Pipeline", "FeatureUnion", "Passthrough", "SkipTransformer", "FunctionTransformer"}
    rows = []
    for s, _ in C.PAIRS:
        for seed in (42, 7, 123):
            x = tp[(tp.sensor == s) & (tp.seed == seed)].iloc[0]
            steps = [e for e in re.findall(r"([A-Z][A-Za-z]+)\(", x.pipeline) if e not in skip]
            steps = list(dict.fromkeys(steps))
            rows.append(" & ".join([NAME[s] + (" T" if s.endswith("_T") else " RH"), str(seed), str(int(x.n_generations)),
                                    str(int(x.n_evaluated)), r" $\rightarrow$ ".join(steps),
                                    "{:.3f}".format(x.R2_hold), "{:.2f}".format(x.MAE_hold)]) + NL)
    write("tab_S2_pipelines", r"@{}llrr p{6.6cm} rr@{}",
          [r"Channel & Seed & Gen. & Evaluated & Selected pipeline (main steps) & $R^2$ & MAE" + NL], rows, longtable=True)


def tests():
    dm = pd.read_csv(C.RES / "testes_robustez.csv")
    lab = lambda a, b: HEAD[a] + " vs.\\ " + HEAD[b]
    g = dm.groupby(["model_a", "model_b"], sort=False)[["sig_1h", "sig_2h", "sig_4h", "sig_6h", "DM_sig_holm"]].sum()
    rows = [lab(a, b) + " & " + " & ".join(str(int(v)) for v in r) + NL for (a, b), r in zip(g.index, g.to_numpy())]
    write("tab_S3_blocklength", "@{}l cccc c@{}",
          [r" & \multicolumn{4}{c}{Moving-block bootstrap, block length} & Diebold--Mariano" + NL, r"\cmidrule(lr){2-5}",
           r"Model pair & 1 h & 2 h & 4 h & 6 h & with Holm correction" + NL], rows)
    rows = []
    for (a, b), sub in dm.groupby(["model_a", "model_b"], sort=False):
        for r in sub.itertuples():
            rows.append(" & ".join([lab(a, b), NAME[r.sensor] + (" T" if r.sensor.endswith("_T") else " RH"),
                                    "{:+.2f}".format(r.dMAE_a_minus_b), "{:+.2f}".format(r.DM_stat),
                                    "{:.3f}".format(r.DM_p), "{:.3f}".format(r.DM_p_holm)]) + NL)
    write("tab_S4_dm", "@{}ll rrrr@{}",
          [r"Model pair (A vs.\ B) & Channel & $\Delta$MAE (A $-$ B) & DM statistic & $p$ & $p$ (Holm)" + NL], rows, longtable=True)


def all_metrics():
    r = pd.concat([pd.read_csv(C.RES / "modelos_protocolos.csv"), pd.read_csv(C.RES / "tpot_protocolos.csv")])
    r = r[r.dataset == "corrigido"]
    protos = [("random", "Random"), ("holdout", "Holdout"), ("blocked", "Blocked"), ("walkfwd", "Walk-fwd")]
    rows = []
    for s, _ in C.PAIRS:
        for m in ("raw", "linear", "quadratic", "RF", "HGB", "TPOT-TS"):
            cells = []
            for p, _ in protos:
                x = r[(r.sensor == s) & (r.model == m) & (r.protocol == p)]
                cells.append("--" if x.empty else "{:.2f} / {:.2f} / {:.2f}".format(x.MAE_mean.item(), x.RMSE_mean.item(), x.R2_mean.item()))
            rows.append(NAME[s] + (" T" if s.endswith("_T") else " RH") + " & " + HEAD[m] + " & " + " & ".join(cells) + NL)
        rows.append(r"\addlinespace")
    write("tab_S5_allmetrics", "@{}ll cccc@{}",
          ["Channel & Model & " + " & ".join(p[1] for p in protos) + NL], rows[:-1], longtable=True)


def asymmetry():
    a = pd.read_csv(C.RES / "assimetria_por_modelo.csv")
    rows, first_rh = [], True
    for s, target in C.PAIRS:
        if target == "umid" and first_rh:
            rows.append(r"\addlinespace"); first_rh = False
        cells = []
        for m in ("linear", "quadratic", "RF", "TPOT-TS"):
            r = a[(a.sensor == s) & (a.model == m)].iloc[0]
            cells.append("{:+.2f} [{:+.2f}, {:+.2f}]".format(r.h, r.h_lo, r.h_hi) + ("$^{*}$" if r.h_sig else ""))
        rows.append(NAME[s] + (" T" if s.endswith("_T") else " RH") + " & " + " & ".join(cells) + NL)
    write("tab_S7_asymmetry", "@{}l cccc@{}", ["Channel & Linear & Quadratic & RF & TPOT-TS" + NL], rows)


def nested_pipelines():
    n = pd.read_csv(C.RES / "tpot_aninhado_resumo.csv")
    rows = []
    for s, _ in C.PAIRS:
        for r in n[n.sensor == s].itertuples():
            rows.append(" & ".join([NAME[s] + (" T" if s.endswith("_T") else " RH"), str(r.fold + 1), str(int(r.n_generations)),
                                    str(int(r.n_evaluated)), r.pipeline.replace(" > ", r" $\rightarrow$ "),
                                    "{:.2f}".format(r.MAE_test)]) + NL)
    write("tab_S8_nested", r"@{}lrrr p{7.4cm} r@{}",
          [r"Channel & Block & Gen. & Evaluated & Pipeline selected on the outer training set (main steps) & MAE" + NL],
          rows, longtable=True)


if __name__ == "__main__":
    alignment(); pipelines(); tests(); all_metrics(); asymmetry(); nested_pipelines()
