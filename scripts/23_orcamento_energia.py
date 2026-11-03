# -*- coding: utf-8 -*-
"""Energy budget of one low-cost end device (ESP32-WROOM-32D + NiceRF LoRa1276F30-915 + 5 sensors + DS1307 RTC +
Solarv1 power board) and the share of that budget taken by the calibration inference.

Every quantity is tagged with its provenance:
  CONFIRMED  - established from the assembly (BOM, photographs, raw data files, assembly records)
  DATASHEET  - nominal value from the manufacturer's documentation
  ASSUMPTION - nominal operating condition adopted; its influence is covered by the scenarios below
Outputs: resultados/energia_no.csv (one row per scenario), MANUSCRITO_NOVO/tables/tab_S10_energy.tex,
tab_S11_energy_scenarios.tex and a summary on the terminal. Nothing here is a measurement: all values are
manufacturer data under the stated assumptions.
"""
import itertools
import math

import pandas as pd

import config as C

TAB = C.ROOT / "MANUSCRITO_NOVO" / "tables"

V_RAIL = 3.3          # V, logic rail (LD1117S33)                                        CONFIRMED (BOM)
V_RADIO = 5.0         # V, radio supply (LoRa1276F30: 3.0-6.5 V, currents specified at 5 V)  DATASHEET / ASSUMPTION (rail)
V_BAT = 7.4           # V, 2S Li-ion pack nominal                                        CONFIRMED (2S1P) / nominal voltage
CYCLE_S = 300.0       # s, batch interval (readings buffered and sent about every 5 min) CONFIRMED (assembly record)
T_ACQ = 3.6           # s, mean interval between readings (raw files: ~167,000 rows / 7 days)  CONFIRMED (data)
N_ACQ = CYCLE_S / T_ACQ                     # readings per batch (~83)
N_CHANNELS = 8                              # calibrated channels (5 T + 3 RH)

# ----------------------------------------------------------------------------- component currents (mA)
ESP32 = {                                   # ESP32-WROOM-32D, Wi-Fi/BT off                             DATASHEET (Espressif)
    "active_240MHz": 50.0,                  # 30-68 mA at 240 MHz; mid value
    "active_80MHz": 25.0,                   # 20-31 mA at 80 MHz
    "light_sleep": 0.8,
}
T_WAKE_ACQ = 0.10                           # s awake per reading when sleeping in between           ASSUMPTION
RADIO = {                                   # NiceRF LoRa1276F30-915 (SX1276) on the LoRa1262F30 pads  CONFIRMED (assembly record)
    "LoRa1276F30-915, +27 dBm": dict(tx=425.0, rx=13.0, sleep=0.01, v=V_RADIO),   # 400-450 mA @ 27 dBm; RX <13 mA; sleep <10 uA  DATASHEET
    "RFM95-class, +17 dBm (comparison)": dict(tx=87.0, rx=10.8, sleep=0.0002, v=V_RAIL),  # Semtech SX1276 datasheet  DATASHEET
}
SENSORS = {                                 # one unit of each type per node                           CONFIRMED (assembly record)
    "BME280": 0.01,                         # 3.6 uA at 1 Hz forced mode; 0.6 mA worst case normal mode  DATASHEET
    "BMP180": 0.01,                         # 5-12 uA at 1 Hz; 0.65 mA peak during conversion
    "DHT22": 0.05 + 1.5 * 0.005 / T_ACQ,    # 40-50 uA standby; 1-1.5 mA for ~5 ms per reading
    "DHT11": 0.10 + 2.5 * 0.020 / T_ACQ,    # 100-150 uA standby; 0.5-2.5 mA for ~20 ms per reading
    "LM35": 0.06,                           # 60 uA quiescent
}
SOLAR_BOARD = {                             # sensors on the Solarv1 board (not logged)               CONFIRMED (BOM) / DATASHEET
    "MCP6001 op-amp": 0.10,
    "TEMT6000 + GUVA-S12SD": 0.05,
}
RTC_DS1307 = 1.5                            # mA, VCC-powered DS1307 module added after the EAGLE design  CONFIRMED (model) / DATASHEET (1.5 mA max)
REGULATORS = {                              # quiescent currents; all three assumed permanently powered  DATASHEET / ASSUMPTION (topology)
    "LD1117 (3.3 V LDO)": (5.0, 5.0),       # (Iq mA, input V)
    "LM2596 (buck)": (5.0, V_BAT),
    "MP1584 (buck)": (0.10, V_BAT),
}
EFF_CONVERSION = 0.80                       # battery -> rail conversion efficiency                    ASSUMPTION
BATTERY_WH = 2 * 2.2 * 3.7                  # 16.3 Wh: 2S1P, two 18650 cells of 2,200 mAh per node      CONFIRMED (assembly record)
USABLE_FRACTION = 0.80                      # usable share of the nominal capacity                     ASSUMPTION
PANEL_W = 40.0                              # W, advertised rating of the panel of each node; not measured  CONFIRMED (nominal only)
BASE = dict(esp32="light sleep between readings", radio="LoRa1276F30-915, +27 dBm", payload="text (70 B/reading)",
            lora="SF7/125 kHz (library default)")   # most plausible configuration (assembly record)

# ----------------------------------------------------------------------------- LoRa air time (Semtech AN1200.13)
def lora_airtime_s(payload_bytes, sf=7, bw_hz=125e3, cr=1, preamble=8, header=True, low_dr_opt=False):
    t_sym = (2 ** sf) / bw_hz
    de = 1 if low_dr_opt else 0
    n_pay = 8 + max(math.ceil((8 * payload_bytes - 4 * sf + 28 + 16 - 20 * (0 if header else 1)) / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preamble + 4.25) * t_sym + n_pay * t_sym

MAX_PAYLOAD = 240                           # bytes per LoRa packet (practical maximum)
PAYLOAD = {                                 # bytes per reading; every reading is transmitted (assembly record)  CONFIRMED (policy) / ASSUMPTION (encoding)
    "text (70 B/reading)": 70,              # CSV-like ASCII -> 25 packets per batch
    "binary (26 B/reading)": 26,            # 10 fields x int16 + 6 time bytes -> 10 packets per batch
}
LORA_PARAMS = {                             # Arduino LoRa library defaults adopted as nominal  ASSUMPTION
    "SF7/125 kHz (library default)": dict(sf=7, bw_hz=125e3),
    "SF10/125 kHz": dict(sf=10, bw_hz=125e3),
}

# ----------------------------------------------------------------------------- inference on the node
T_INF_QUAD = 2e-6                           # s per quadratic evaluation (10 coef. + scaling) on the ESP32  ASSUMPTION (upper bound; not measured)
P_INF = ESP32["active_240MHz"] * V_RAIL / 1000.0   # W while computing


def mJ(i_mA, v, t_s):
    return i_mA / 1000.0 * v * t_s * 1000.0


rows = []
for (esp_mode, radio_name, payload_name, lora_name) in itertools.product(
        ["always on, 240 MHz", "always on, 80 MHz", "light sleep between readings"], RADIO, PAYLOAD, LORA_PARAMS):
    if esp_mode == "always on, 240 MHz":
        e_esp = mJ(ESP32["active_240MHz"], V_RAIL, CYCLE_S)
    elif esp_mode == "always on, 80 MHz":
        e_esp = mJ(ESP32["active_80MHz"], V_RAIL, CYCLE_S)
    else:
        t_awake = N_ACQ * T_WAKE_ACQ
        e_esp = mJ(ESP32["active_80MHz"], V_RAIL, t_awake) + mJ(ESP32["light_sleep"], V_RAIL, CYCLE_S - t_awake)
    r = RADIO[radio_name]
    bytes_cycle = N_ACQ * PAYLOAD[payload_name]
    n_pkt = math.ceil(bytes_cycle / MAX_PAYLOAD)
    last = bytes_cycle - (n_pkt - 1) * MAX_PAYLOAD
    t_air = (n_pkt - 1) * lora_airtime_s(MAX_PAYLOAD, **LORA_PARAMS[lora_name]) + lora_airtime_s(last, **LORA_PARAMS[lora_name])
    e_radio = mJ(r["tx"], r["v"], t_air) + mJ(r["sleep"], r["v"], CYCLE_S - t_air)
    e_sens = sum(mJ(i, V_RAIL, CYCLE_S) for i in SENSORS.values())
    e_solar_sens = sum(mJ(i, V_RAIL, CYCLE_S) for i in SOLAR_BOARD.values())
    e_rtc = mJ(RTC_DS1307, V_RAIL, CYCLE_S)
    e_reg = sum(mJ(iq, v, CYCLE_S) for iq, v in REGULATORS.values())
    e_inf_record = N_CHANNELS * P_INF * T_INF_QUAD * 1000.0
    e_inf_reading = N_ACQ * e_inf_record
    e_rail = e_esp + e_radio + e_sens + e_solar_sens + e_rtc
    e_conv = e_rail / EFF_CONVERSION - e_rail
    e_total = e_rail + e_conv + e_reg
    wh_day = e_total / 1000.0 * 288 / 3600.0
    rows.append(dict(esp32=esp_mode, radio=radio_name, payload=payload_name, lora=lora_name,
                     packets_per_cycle=n_pkt, airtime_s=round(t_air, 3),
                     ESP32_mJ=e_esp, radio_mJ=e_radio, sensors_mJ=e_sens, solar_board_sensors_mJ=e_solar_sens,
                     RTC_mJ=e_rtc, regulators_mJ=e_reg, conversion_mJ=e_conv, total_mJ=e_total, total_Wh_day=wh_day,
                     inference_per_record_mJ=e_inf_record, inference_per_reading_mJ=e_inf_reading,
                     inference_share_ppm=e_inf_reading / e_total * 1e6,
                     autonomy_days=BATTERY_WH * USABLE_FRACTION / wh_day,
                     full_sun_min_per_day=wh_day / PANEL_W * 60,
                     sensors_share_pct=e_sens / e_total * 100, radio_share_pct=e_radio / e_total * 100,
                     ESP32_share_pct=e_esp / e_total * 100, regulators_share_pct=e_reg / e_total * 100))

df = pd.DataFrame(rows)
df.to_csv(C.RES / "energia_no.csv", index=False, float_format="%.4f")

b = df[(df.esp32 == BASE["esp32"]) & (df.radio == BASE["radio"]) & (df.payload == BASE["payload"]) & (df.lora == BASE["lora"])].iloc[0]

# ----------------------------------------------------------------------------- LaTeX tables for the Supplementary Material
comp = [
    ("Radio, NiceRF LoRa1276F30-915: %d packets, %.1f s on air at +27 dBm per batch; sleep otherwise" % (b.packets_per_cycle, b.airtime_s),
     "425 mA at 5 V (TX); $<$10 $\\mu$A (sleep)", "datasheet; packets, SF7/125 kHz assumed", b.radio_mJ),
    ("Regulators LD1117, LM2596 and MP1584, quiescent, permanently powered",
     "5, 5 and 0.1 mA", "datasheet; topology assumed", b.regulators_mJ),
    ("ESP32-WROOM-32D: 0.1 s awake per reading at 80 MHz, light sleep otherwise, Wi-Fi and Bluetooth off",
     "25 mA (awake); 0.8 mA (light sleep)", "datasheet; duty cycle assumed", b.ESP32_mJ),
    ("Real-time clock DS1307", "1.5 mA", "datasheet (maximum)", b.RTC_mJ),
    ("Five environmental sensors (BME280, BMP180, DHT11, DHT22, LM35), one reading every 3.6 s",
     "0.01--0.11 mA each (average)", "datasheet", b.sensors_mJ),
    ("Power-board sensors (TEMT6000, GUVA-S12SD, MCP6001), not logged", "0.15 mA", "datasheet", b.solar_board_sensors_mJ),
    ("Conversion losses, battery to the 3.3 V and 5 V rails", "80\\% efficiency", "assumed", b.conversion_mJ),
]
lines = ["\\begin{tabular}{@{}p{6.6cm}p{3.0cm}p{3.0cm}rr@{}}", "\\toprule",
         "Component and operating condition & Current & Provenance & J per batch & Share (\\%) \\\\", "\\midrule"]
for name, cur, prov, e in comp:
    lines.append("%s & %s & %s & %.2f & %.1f \\\\" % (name, cur, prov, e / 1000.0, e / b.total_mJ * 100))
lines += ["\\midrule",
          "Total per node and 5-min batch & & & %.2f & 100 \\\\" % (b.total_mJ / 1000.0),
          "Quadratic calibration of 8 channels evaluated on every reading (2 $\\mu$s per evaluation at 50 mA, assumed upper bound) & & assumed & %.2f mJ & %.1f ppm \\\\" % (
              b.inference_per_reading_mJ, b.inference_share_ppm),
          "\\bottomrule", "\\end{tabular}"]
(TAB / "tab_S10_energy.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

sc = df[df.radio == BASE["radio"]].copy()
SHORT = {"always on, 240 MHz": "Always on, 240 MHz", "always on, 80 MHz": "Always on, 80 MHz",
         "light sleep between readings": "Light sleep"}
lines = ["\\begin{tabular}{@{}lllrrrr@{}}", "\\toprule",
         "ESP32 & Encoding (packets per batch) & LoRa & Air time (s) & Wh day$^{-1}$ & Autonomy (d) & Inference (ppm) \\\\", "\\midrule"]
for _, r in sc.iterrows():
    lines.append("%s & %s (%d) & %s & %.1f & %.2f & %.1f & %.1f \\\\" % (
        SHORT[r.esp32], r.payload.split(" (")[0], r.packets_per_cycle, r.lora.replace(" (library default)", "").replace("/125 kHz", ""),
        r.airtime_s, r.total_Wh_day, r.autonomy_days, r.inference_share_ppm))
lines += ["\\bottomrule", "\\end{tabular}"]
(TAB / "tab_S11_energy_scenarios.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
cols = ["esp32", "radio", "payload", "lora", "packets_per_cycle", "airtime_s", "total_Wh_day", "autonomy_days",
        "ESP32_share_pct", "radio_share_pct", "sensors_share_pct", "regulators_share_pct", "inference_share_ppm"]
print(df[cols].round(2).to_string(index=False))
print("\nBASE CASE:", BASE)
for k in ["ESP32_mJ", "radio_mJ", "sensors_mJ", "solar_board_sensors_mJ", "RTC_mJ", "regulators_mJ", "conversion_mJ", "total_mJ"]:
    print("  %-23s %9.1f mJ per batch  (%5.1f %%)" % (k, b[k], b[k] / b["total_mJ"] * 100))
print("  per day: %.2f Wh per node | battery %.1f Wh nominal, %.0f %% usable -> %.1f days without sun | panel %.0f W nominal: %.1f min of full sun per day" % (
    b.total_Wh_day, BATTERY_WH, USABLE_FRACTION * 100, b.autonomy_days, PANEL_W, b.full_sun_min_per_day))
print("  inference (quadratic, 8 channels, every reading): %.3f mJ per batch = %.1f ppm of the node budget" % (
    b.inference_per_reading_mJ, b.inference_share_ppm))
# --- gateway: same board and LoRa module as the end device (ESP32 always on, radio in continuous receive), feeding a
#     Raspberry Pi 3 Model B (typical bare-board active current 400 mA at 5 V, raspberrypi.com documentation)   CONFIRMED (assembly record) / DATASHEET
P_PI = 0.400 * 5.0
P_GW_BOARD = (ESP32["active_240MHz"] * V_RAIL + RADIO[BASE["radio"]]["rx"] * V_RADIO) / 1000.0 / EFF_CONVERSION
WH_GATEWAY = (P_PI + P_GW_BOARD) * 24
print("  gateway: Raspberry Pi 3 B %.2f W + receiver board %.2f W = %.1f Wh/day = %.1f x this node" % (
    P_PI, P_GW_BOARD, WH_GATEWAY, WH_GATEWAY / b.total_Wh_day))
print("\nscenario range (NiceRF radio): %.2f-%.2f Wh/day; autonomy %.1f-%.1f days; inference %.1f-%.1f ppm" % (
    sc.total_Wh_day.min(), sc.total_Wh_day.max(), sc.autonomy_days.min(), sc.autonomy_days.max(),
    sc.inference_share_ppm.min(), sc.inference_share_ppm.max()))
print("saved", C.RES / "energia_no.csv", TAB / "tab_S10_energy.tex", TAB / "tab_S11_energy_scenarios.tex")
