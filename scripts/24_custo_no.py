# -*- coding: utf-8 -*-
"""Component cost of one end device, from the traceable quotation compiled on 02/11/2026
(dados/cotacao_componentes_2026.xlsx: public retail prices of 2 Nov 2026, with a link per item).
The spreadsheet quotes three nodes; this script aggregates per node, converts USD at the PTAX selling rate of
30/10/2026 used in the spreadsheet, and writes MANUSCRITO_NOVO/tables/tab_S9_cost.tex and resultados/custo_no.csv.
Prices are 2026 references, not the historical acquisition cost of the 2018 assembly. Excluded from the quotation
(no price): fabrication of the Solarv1 PCB, assembly, the 2S charger, shipping and import taxes, and two parts out of
stock (47 uF TDK capacitor, 2120-CB protection IC).
"""
import pandas as pd

import config as C

XLSX = C.ROOT / "dados" / "cotacao_componentes_2026.xlsx"
TAB = C.ROOT / "MANUSCRITO_NOVO" / "tables"
N_NODES = 3

prem = pd.read_excel(XLSX, sheet_name="Premissas energia", header=None)
PTAX = float(prem[prem[0].astype(str).str.contains("PTAX")].iloc[0, 1])

q = pd.read_excel(XLSX, sheet_name="Cotacao por componente", header=5)
q = q.rename(columns={q.columns[0]: "grupo", q.columns[2]: "item", q.columns[6]: "status", q.columns[7]: "moeda",
                      q.columns[11]: "subtotal"})
q = q[q.subtotal.notna()].copy()
q["brl_3"] = q.apply(lambda r: r.subtotal * PTAX if r.moeda == "USD" else r.subtotal, axis=1)

groups = {
    "Estufa": "Sensor board (ESP32-WROOM-32D, BME280, LoRa1276F30-915 module with antenna, passives, connectors)",
    "Solarv1": "Power board (LM2596, MP1584, LD1117, 8205A MOSFETs, TEMT6000, GUVA-S12SD, MCP6001, passives)",
    "Sensores": "Other sensors (BMP180, DHT11, DHT22, LM35 modules)",
    "Infraestrutura": "Sensor-board PCB fabrication and DS1307 RTC module",
    "Solar hipotético": "Power supply (40 W panel kit with controller; two 18650 cells of 2,200 mAh)",
}
rows = []
for g, label in groups.items():
    sub = q[q.grupo == g]
    rows.append(dict(group=label, brl_per_node=sub.brl_3.sum() / N_NODES, n_items=len(sub)))
cost = pd.DataFrame(rows)
cost["usd_per_node"] = cost.brl_per_node / PTAX
total_brl = cost.brl_per_node.sum()
elec_brl = cost[~cost.group.str.startswith("Power supply")].brl_per_node.sum()
radio = q[q.item.astype(str).str.contains("LoRa1276F30")].brl_3.sum() / N_NODES
cost.to_csv(C.RES / "custo_no.csv", index=False, float_format="%.2f")

# --- gateway: a receiver built on the same board and LoRa module as the end devices (assembly record), plus a
#     Raspberry Pi 3 Model B at its launch list price of US$35 (raspberrypi.com, 29 Feb 2016; not re-quoted in 2026)
gw_board_brl = cost.loc[0, "brl_per_node"]
pi_usd = 35.0
gw_brl = gw_board_brl + pi_usd * PTAX
system_brl = N_NODES * total_brl + gw_brl

lines = ["\\begin{tabular}{@{}p{10.5cm}rr@{}}", "\\toprule", "Item & R\\$ & US\\$ \\\\", "\\midrule",
         "\\multicolumn{3}{@{}l}{\\emph{One end device}} \\\\"]
for _, r in cost.iterrows():
    lines.append("\\quad %s & %.0f & %.0f \\\\" % (r.group, r.brl_per_node, r.usd_per_node))
lines += ["\\quad Electronics and sensors (excluding power supply) & %.0f & %.0f \\\\" % (elec_brl, elec_brl / PTAX),
          "\\quad Total per end device & %.0f & %.0f \\\\" % (total_brl, total_brl / PTAX),
          "\\midrule", "\\multicolumn{3}{@{}l}{\\emph{Gateway}} \\\\",
          "\\quad LoRa receiver (same board and module as the end device) & %.0f & %.0f \\\\" % (gw_board_brl, gw_board_brl / PTAX),
          "\\quad Raspberry Pi~3 Model~B (launch list price) & %.0f & %.0f \\\\" % (pi_usd * PTAX, pi_usd),
          "\\midrule", "Three end devices and the gateway & %.0f & %.0f \\\\" % (system_brl, system_brl / PTAX),
          "\\bottomrule", "\\end{tabular}"]
(TAB / "tab_S9_cost.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(cost.round(1).to_string(index=False))
print("PTAX %.4f | total per node R$ %.1f = US$ %.1f | electronics+sensors R$ %.1f = US$ %.1f | radio kit R$ %.1f = US$ %.1f (%.0f %% of electronics)" % (
    PTAX, total_brl, total_brl / PTAX, elec_brl, elec_brl / PTAX, radio, radio / PTAX, radio / elec_brl * 100))
print("gateway R$ %.1f = US$ %.1f | system (3 nodes + gateway) R$ %.1f = US$ %.1f" % (gw_brl, gw_brl / PTAX, system_brl, system_brl / PTAX))
print("saved", TAB / "tab_S9_cost.tex")
