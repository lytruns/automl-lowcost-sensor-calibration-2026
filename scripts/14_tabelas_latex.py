# -*- coding: utf-8 -*-
"""Generate the manuscript tables (LaTeX, booktabs) directly from the result CSVs,
so that no number is typed by hand.  Only the clock-converted dataset is used.
Output: MANUSCRITO_NOVO/tables/*.tex
"""
import re

import numpy as np
import pandas as pd

import config as C

OUT = C.ROOT / "MANUSCRITO_NOVO" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
NAME = {s: s.split("_")[0] for s, _ in C.PAIRS}
MODELS = ["linear", "quadratic", "RF", "TPOT-TS"]
HEAD = {"raw": "Raw", "linear": "Linear", "quadratic": "Quadratic", "RF": "RF", "TPOT-TS": "TPOT-TS"}
UNIT_T = r"Air temperature ($^\circ$C)"
UNIT_RH = r"Relative humidity (\% RH)"
NL = r" \\"


def write(name, header_lines, body_lines, colspec):
    # signed numbers in math mode, so that the minus sign is typeset correctly
    body_lines = [re.sub(r"(?<![\d\-$])([+-]\d+(?:\.\d+)?)", r"$\1$", ln) for ln in body_lines]
    body_lines = [re.sub(r"\$[+-](0\.0+)\$", r"$\1$", ln) for ln in body_lines]      # no signed zero
    text = "\n".join([rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *header_lines, r"\midrule",
                      *body_lines, r"\bottomrule", r"\end{tabular}", ""])
    (OUT / f"{name}.tex").write_text(text, encoding="utf-8")
    print("wrote", name)


def block(rows_t, rows_rh, ncol, unit_t=UNIT_T, unit_rh=UNIT_RH):
    span = lambda u: rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{u}}}}}" + NL
    return [span(unit_t), *rows_t, r"\midrule", span(unit_rh), *rows_rh]


def bold_min(vals, fmt):
    txt = [fmt.format(v) for v in vals]
    best = fmt.format(np.nanmin(vals))                      # ties at the printed precision are all bold
    return [(r"\textbf{" + t + "}") if t == best else t for t in txt]


def star(lo, hi):
    return "$^{*}$" if (lo > 0 or hi < 0) else ""


def main():
    res = pd.concat([pd.read_csv(C.RES / "modelos_protocolos.csv"), pd.read_csv(C.RES / "tpot_protocolos.csv")])
    res = res[res.dataset == "corrigido"]
    heads = " & ".join(HEAD[m] for m in MODELS)

    # ---- blocked-CV performance ----
    b = res[res.protocol == "blocked"]
    rows = {"temp": [], "umid": []}
    for s, t in C.PAIRS:
        g = b[b.sensor == s].set_index("model")
        mae = bold_min([g.loc[m, "MAE_mean"] for m in MODELS], "{:.2f}")
        cells = ["{:.2f}".format(g.loc["raw", "MAE_mean"])]
        cells += [v + r" $\pm$ " + "{:.2f}".format(g.loc[m, "MAE_sd"]) for v, m in zip(mae, MODELS)]
        cells += ["{:.2f}".format(g.loc[m, "R2_mean"]) for m in MODELS]
        rows[t].append(NAME[s] + " & " + " & ".join(cells) + NL)
    write("tab_performance",
          [r" & \multicolumn{5}{c}{MAE (mean $\pm$ s.d. across blocks)} & \multicolumn{4}{c}{$R^2$}" + NL,
           r"\cmidrule(lr){2-6}\cmidrule(l){7-10}", "Sensor & Raw & " + heads + " & " + heads + NL],
          block(rows["temp"], rows["umid"], 10), "@{}l c cccc cccc@{}")

    # ---- validation protocols (temporal leakage) ----
    protos = ["random", "holdout", "blocked", "walkfwd"]
    rows = {"temp": [], "umid": []}
    for t in ("temp", "umid"):
        g = res[(res.target == t) & res.model.isin(MODELS)].groupby(["model", "protocol"]).MAE_mean.mean().unstack()
        for m in MODELS:
            # TPOT-TS is reported only where the architecture search never saw the test data
            # (temporal holdout; blocked CV with the nested search of script 18)
            infl = 100 * (g.loc[m, "blocked"] / g.loc[m, "random"] - 1)
            cells = ["--" if np.isnan(g.loc[m, p]) else "{:.2f}".format(g.loc[m, p]) for p in protos]
            rows[t].append(HEAD[m] + " & " + " & ".join(cells)
                           + " & " + ("--" if np.isnan(infl) else "{:+.0f}".format(infl)) + NL)
    write("tab_protocols",
          [r"Model & Random split & Temporal holdout & Blocked CV & Walk-forward & Blocked vs.\ random (\%)" + NL],
          block(rows["temp"], rows["umid"], 6, r"Air temperature, MAE ($^\circ$C), mean over five sensors",
                r"Relative humidity, MAE (\% RH), mean over three sensors"), "@{}l cccc c@{}")

    # ---- paired tests and equivalence ----
    dm = pd.read_csv(C.RES / "testes_robustez.csv")
    eq = pd.read_csv(C.RES / "equivalencia.csv")
    rows = {"temp": [], "umid": []}
    for s, t in C.PAIRS:
        cells = []
        for a, other in (("quadratic", "RF"), ("quadratic", "TPOT-TS")):
            d = dm[(dm.model_a == a) & (dm.model_b == other) & (dm.sensor == s)].iloc[0]
            e = eq[(eq.comparison == a + " - " + other) & (eq.sensor == s)].iloc[0]
            cells.append("{:+.2f}".format(d.dMAE_a_minus_b) + ("$^{*}$" if d.DM_sig_holm else ""))
            cells.append("[{:+.2f}, {:+.2f}]".format(e.ci90_lo, e.ci90_hi))
        rows[t].append(NAME[s] + " & " + " & ".join(cells) + NL)
    write("tab_tests",
          [r" & \multicolumn{2}{c}{Quadratic $-$ RF} & \multicolumn{2}{c}{Quadratic $-$ TPOT-TS}" + NL,
           r"\cmidrule(lr){2-3}\cmidrule(l){4-5}", r"Sensor & $\Delta$MAE & 90\% CI & $\Delta$MAE & 90\% CI" + NL],
          block(rows["temp"], rows["umid"], 5), "@{}l cc cc@{}")

    # ---- polynomial degree ----
    g = pd.read_csv(C.RES / "grau_polinomio.csv")
    g = g[g.inputs == "3 replicas"]
    lines = []
    for deg in range(1, 6):
        vals = []
        for t in ("temp", "umid"):
            x = g[(g.degree == deg) & (g.target == t)]
            vals += [x.MAE_insample.mean(), x.MAE_blocked.mean(), x.MAE_walkfwd.mean()]
        lines.append("{} & {} & ".format(deg, int(g[g.degree == deg].n_coef.iloc[0]))
                     + " & ".join("{:.2f}".format(v) for v in vals) + NL)
    write("tab_degree",
          [r" & & \multicolumn{3}{c}{Air temperature, MAE ($^\circ$C)} & \multicolumn{3}{c}{Relative humidity, MAE (\% RH)}" + NL,
           r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
           "Degree & Coefficients & In-sample & Blocked CV & Walk-forward & In-sample & Blocked CV & Walk-forward" + NL],
          lines, "@{}cc ccc ccc@{}")

    # ---- replica ablation ----
    ab = pd.read_csv(C.RES / "ablacao_replicas.csv")
    ab = ab[ab.model.isin(["linear", "quadratic", "RF"])].copy()
    ab["target"] = np.where(ab.sensor.str.endswith("_T"), "temp", "umid")
    p = ab.groupby(["target", "replicas", "model"]).MAE_mean.mean().unstack()
    lines = ["{} & ".format(k) + " & ".join("{:.2f}".format(p.loc[(t, k), m]) for t in ("temp", "umid")
                                            for m in ("linear", "quadratic", "RF")) + NL for k in (1, 2, 3)]
    write("tab_ablation",
          [r" & \multicolumn{3}{c}{Air temperature, MAE ($^\circ$C)} & \multicolumn{3}{c}{Relative humidity, MAE (\% RH)}" + NL,
           r"\cmidrule(lr){2-4}\cmidrule(l){5-7}", "Units per sensor type & Linear & Quadratic & RF & Linear & Quadratic & RF" + NL],
          lines, "@{}c ccc ccc@{}")

    # ---- inference footprint ----
    inf = pd.read_csv(C.RES / "custo_inferencia.csv")
    lines = []
    for m in MODELS:
        x = inf[inf.model == m]
        if m == "linear":
            par = "4 coefficients"
        elif m == "quadratic":
            par = "10 coefficients"
        elif m == "RF":
            par = "{:,.0f} tree nodes".format(x.parameters_or_nodes.median())
        else:
            par = "pipeline-dependent"
        if m == "TPOT-TS":
            size = "{:,.0f}--{:,.0f}".format(x.size_kB.min(), x.size_kB.max())
            lat = "{:.1f}--{:.1f}".format(x.latency_ms.min(), x.latency_ms.max())
        else:
            size = "{:,.1f}".format(x.size_kB.median())
            lat = "{:.2f}".format(x.latency_ms.median())
        lines.append(" & ".join([HEAD[m], par, size, lat]) + NL)
    write("tab_footprint", ["Model & Parameters & Serialized size (kB) & Latency per sample (ms)" + NL],
          lines, "@{}l l c c@{}")

    # ---- TPOT search stability ----
    tp = pd.read_csv(C.RES / "tpot_resumo_buscas.csv")
    tp = tp[tp.dataset == "corrigido"]
    rows = {"temp": [], "umid": []}
    for s, t in C.PAIRS:
        x = tp[tp.sensor == s].set_index("seed")
        rows[t].append(NAME[s] + " & " + " & ".join("{:.3f}".format(x.loc[k, "R2_hold"]) for k in (42, 7, 123))
                       + " & {:.3f} & ".format(x.R2_hold.max() - x.R2_hold.min())
                       + " / ".join("{:.2f}".format(x.loc[k, "MAE_hold"]) for k in (42, 7, 123))
                       + " & {:+.2f} to {:+.2f}".format(x.overfit_gap.min(), x.overfit_gap.max()) + NL)
    write("tab_tpot_seeds",
          [r" & \multicolumn{3}{c}{$R^2$ (temporal holdout)} & & &" + NL, r"\cmidrule(lr){2-4}",
           r"Sensor & Seed 42 & Seed 7 & Seed 123 & Range & MAE (42 / 7 / 123) & Train$-$validation gap" + NL],
          block(rows["temp"], rows["umid"], 7), "@{}l ccc c c c@{}")

    # ---- dynamic response ----
    dy = pd.read_csv(C.RES / "resposta_dinamica.csv")
    dy = dy[dy.support.isin(["A_5min", "B_1min"])]
    rows = {"temp": [], "umid": []}
    for s, t in C.PAIRS:
        x = dy[dy.sensor == s].iloc[0]
        rows[t].append(NAME[s]
                       + r" & {:+.1f} $\pm$ {:.1f} & {:+.1f} $\pm$ {:.1f}".format(x.delay_day, x.delay_day_sd, x.delay_night, x.delay_night_sd)
                       + " & {:+.1f} [{:+.1f}, {:+.1f}]".format(x.tau_min, x.tau_lo, x.tau_hi) + star(x.tau_lo, x.tau_hi)
                       + " & {:+.2f} [{:+.2f}, {:+.2f}]".format(x.h_units, x.h_lo, x.h_hi) + star(x.h_lo, x.h_hi) + NL)
    write("tab_dynamic",
          [r" & \multicolumn{2}{c}{Relative time shift (min)} & Rate-dependent term & Heating--cooling offset" + NL,
           r"\cmidrule(lr){2-3}", r"Sensor & Day & Night & $\tau$ (min) [95\% CI] & $h$ [95\% CI]" + NL],
          block(rows["temp"], rows["umid"], 5), "@{}l cc c c@{}")

    # ---- RH temporal support (supplementary) ----
    su = pd.read_csv(C.RES / "ur_suporte_temporal.csv")
    lines = [" & ".join([NAME[r.sensor], HEAD[r.model], "{:.2f}".format(r.MAE_A_5min), "{:.2f}".format(r.MAE_B_1min),
                         "{:.2f}".format(r.MAE_C_10s),
                         "{:+.2f} [{:+.2f}, {:+.2f}]".format(r.dMAE_A_minus_C_10s, r.ci_lo_C_10s, r.ci_hi_C_10s)]) + NL
             for r in su.itertuples()]
    write("tab_rh_support",
          [r"Sensor & Model & 5-min mean & 1-min mean & Nearest 10-s reading & 5-min $-$ 10-s [95\% CI]" + NL],
          lines, "@{}ll ccc c@{}")


if __name__ == "__main__":
    main()
