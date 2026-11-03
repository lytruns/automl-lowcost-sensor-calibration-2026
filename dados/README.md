# Data

## Structure

| Folder / file | Content |
|---|---|
| `brutos/` | Raw files, unmodified. SHA-256 hashes in `SHA256SUMS.txt`. |
| `brutos/Dados da estação estufa.xlsx` | Reference: Vaisala HMP45C probe + Campbell Scientific CR23X data logger. 5-min records from 11 Nov 2018 12:00 to 15 Nov 2018 10:20 (1,133 rows). Temperature = mean of the interval; RH = instantaneous sample. Timestamp at the **end** of the interval (Campbell convention), in **Brasília standard time (UTC−3)**. Columns `temp` (°C), `umid` (% RH), wind speed and direction. |
| `brutos/Dispositivo A/B/C.xlsx` | Low-cost system: three units (A, B, C), each with BME280, BMP180, DHT11, DHT22 and LM35. One reading every ~3.6 s (about 167,000 rows per unit), 8 to 15 Nov 2018. Time recorded by the unit's RTC in the columns `Hora, Minuto, Segundo, Ano, MesT, Dia`, in **daylight-saving time (UTC−2)**. |
| `processados/dataset_original_5min.csv` | 5-min dataset with the timestamps **as recorded**. Kept for comparison only. |
| `processados/dataset_corrigido_5min.csv` | 5-min dataset with the **60-min clock correction**. This is the dataset used in the article. |
| `processados/dataset_info.json` | Number of rows, period, ranges of T and RH, discarded rows. |
| `cotacao_componentes_2026.xlsx` | Component quotation of 2 November 2026 (public retail prices, one supplier link per item; PTAX rate of 30 Oct 2026) used by `scripts/24_custo_no.py`. |

## Clock correction (daylight-saving time)

Brazilian daylight-saving time 2018/19 started at 00:00 on **4 November 2018**. The RTCs of the low-cost units were
set to civil time (UTC−2) before the campaign, whereas the CR23X stayed on standard time (UTC−3). The low-cost
timestamps are therefore 60 min ahead of the reference.

The correction is **fixed a priori at 60 min** (known physical cause), not tuned to the data. Three independent
checks, all reproduced by `scripts/02_diagnostico_relogio.py` on the training portion only (first 80 %), support it:

1. **Cross-correlation of first differences** between each of the 24 low-cost channels and the reference: about 0.11
   with no shift; narrow peak at **59.2 min** (median; IQR 57.9–61.1). The peak is the same for temperature and
   humidity and for sensors with a 1-s response time, which rules out thermal inertia as the cause.
2. **Sunrise** (22.85° S, 48.43° W): 05:22 in standard time, 06:22 in daylight-saving time. The reference starts to
   warm at 05:50–05:55 (consistent only with standard time); the low-cost system at 06:40–07:00 on its own clock
   (consistent only with daylight-saving time).
3. **Sensitivity**: the error of a quadratic calibration (blocked CV within the training portion) is minimal for
   shifts of 50–65 min for every sensor–variable pair.

After the correction, the residual lag between calibrated prediction and reference is 0 min (±5 min, the resolution
of the data).

## Construction of the 5-min datasets (`scripts/01_construir_dataset.py`)

1. Raw readings (~3.6 s) → 10-s grid (mean; gaps ≤ 1 min linearly interpolated).
2. For each reference timestamp `T`, mean of the low-cost readings in `[T + offset − 5 min, T + offset)`, that is,
   the same end-of-interval convention as the CR23X (`offset` = 0 for the original dataset, 3600 s for the
   corrected one).
3. Rows with any missing value are discarded (14 of 1,133 → **1,119 records**). No other filtering or smoothing is
   applied.

Columns: `A_|B_|C_` + `BME280_T, BMP180_T, DHT22_T, LM35_T, DHT11_T` (°C) and `BME280_U, DHT22_U, DHT11_U` (% RH);
`temp` (°C) and `umid` (% RH) are the reference.

## Scope of the data

- Collocation window of **3.9 days** (about 94 h) in one commercial greenhouse, one shelter and one season.
- The 3D-printed shelter heats under direct sun: raw bias of up to +10–13 °C at midday. The calibration corrects
  mainly this radiative effect, which is specific to the shelter and the site.
- The reference RH is an instantaneous sample, whereas the low-cost RH is a 5-min mean.
