# When is AutoML worth it? Data and code

Data and analysis pipeline for the manuscript

> **When is AutoML worth it? Temporal validation and metrological assessment of low-cost greenhouse sensor calibration**
> L. A. G. Rocha, A. L. dos Santos Safre, R. L. de Sousa, I. A. Guerrini. Submitted to *Measurement* (Elsevier/IMEKO), 2026.

Five low-cost air-temperature and relative-humidity sensor types (BMP180, BME280, DHT11, DHT22, LM35; three units
each) were collocated with a Vaisala HMP45C reference probe in a commercial greenhouse (Botucatu, SP, Brazil,
11-15 November 2018). The repository contains the raw records, the 5-min datasets, every script of the analysis
(calibration models, validation protocols, nested AutoML search, statistical tests, equivalence analysis, replica
ablation, dynamic response, energy and cost budgets) and the result files from which the tables and figures of the
article are generated.

## Repository structure

```
dados/
  brutos/                     Raw files, unmodified (SHA-256 in dados/SHA256SUMS.txt)
    Dados da estação estufa.xlsx   Reference: Vaisala HMP45C + Campbell CR23X, 5-min records (1,133 rows)
    Dispositivo A|B|C.xlsx         Low-cost units A, B, C: one reading every ~3.6 s per sensor (~167,000 rows each)
  processados/
    dataset_original_5min.csv      5-min dataset with the timestamps as recorded (kept for comparison only)
    dataset_corrigido_5min.csv     5-min dataset with the 60-min clock correction: the dataset used in the article
    dataset_info.json              Rows, period, ranges, discarded rows
  cotacao_real_componentes_3_nos_2026.xlsx   Component quotation (2 Nov 2026) used by scripts/24_custo_no.py
  README.md                   Data description and the clock-correction evidence (in Portuguese)
scripts/                      Numbered pipeline; shared settings in scripts/config.py
resultados/                   CSV results, TPOT pipelines (JSON + pickled scikit-learn objects), figures
MANUSCRITO_NOVO/tables, figures   Output folders of scripts 14, 15 and 17 (article tables and figures)
```

### Clock correction

The real-time clocks of the low-cost units were set to Brazilian civil time after the start of daylight-saving time
(4 November 2018, UTC-2), whereas the reference logger remained on standard time (UTC-3). The low-cost timestamps are
therefore 60 min ahead of the reference. The correction is fixed a priori from this physical cause and verified
empirically on the training portion only (`scripts/02_diagnostico_relogio.py`; see `dados/README.md`).

## Pipeline

| Script | Purpose | Main outputs |
|---|---|---|
| `01_construir_dataset.py` | Raw files to 5-min datasets (original and clock-corrected) | `dados/processados/*.csv` |
| `02_diagnostico_relogio.py` | Three independent checks of the 60-min offset (training portion only) | `resultados/relogio_*.csv` |
| `03_modelos_protocolos.py` | Linear, quadratic, random forest and HGB under four validation protocols | `modelos_protocolos.csv` |
| `04_tpot.py` | TPOT searches (fixed temporal split), three seeds on the corrected dataset | `resultados/tpot/` |
| `05_analises_complementares.py` | Replica ablation, paired block bootstrap, dynamic error, inference footprint | `ablacao_replicas.csv`, `bootstrap_pares.csv`, ... |
| `06_tpot_protocolos.py` | TPOT pipelines re-evaluated under the protocols; stability across seeds | `tpot_protocolos.csv`, `tpot_resumo_buscas.csv` |
| `07_figuras.py` | Working figures from the CSV files | `resultados/figuras/` |
| `08_grau_polinomio.py` | Polynomial degree 1-5: in-sample versus out-of-sample error | `grau_polinomio.csv` |
| `09_robustez_testes.py` | Block bootstrap (1-6 h blocks), Diebold-Mariano test with Holm correction | `testes_robustez.csv` |
| `10_equivalencia.py` | Practical equivalence of quadratic versus RF/TPOT within the reference accuracy specification | `equivalencia.csv` |
| `12_atraso_por_janela.py` | Lag by 8-h windows (timestamps as recorded) | `atraso_por_janela.csv` |
| `13_fechamento_dinamica.py` | Temporal support of RH and dynamic response (rate term and heating-cooling offset) | `ur_suporte_temporal.csv`, `resposta_dinamica.csv` |
| `14_tabelas_latex.py` | Article tables (LaTeX) from the CSV files | `MANUSCRITO_NOVO/tables/` |
| `15_figuras_artigo.py` | Article figures | `MANUSCRITO_NOVO/figures/` |
| `17_suplementar.py` | Supplementary figure and tables | `MANUSCRITO_NOVO/` |
| `18_tpot_aninhado.py` | Nested evaluation of the AutoML procedure under blocked cross-validation | `resultados/tpot_aninhado/` |
| `19_assimetria_por_modelo.py` | Heating-cooling offset and rate term on the out-of-fold residuals of each model family | `assimetria_*.csv` |
| `20_integrar_aninhado.py` | Integrates the nested TPOT evaluation into the result files used by the article | `predicoes_oof.csv`, `tpot_protocolos.csv` |
| `23_orcamento_energia.py` | Energy budget of one end device from manufacturer data | `energia_no.csv`, Tables S10-S11 |
| `24_custo_no.py` | Component cost of one end device from the 2026 quotation | `custo_no.csv`, Table S9 |

Execution order: 01, 02, 03, 04 (hours; resumable), 05, 06, 18, 20, 08, 09, 10, 13, 19, 12, 23, 24, then 07, 14, 15, 17.
All result files are included, so any script after 04 can be run directly on the provided results.

### Validation protocols

- **random**: 80/20 random split, 10 repetitions; reported only to quantify temporal leakage.
- **holdout**: first 80% for training, last 20% for testing.
- **blocked**: four contiguous blocks of about one day with a 1-h embargo; primary protocol of the article.
- **walkfwd**: expanding window, five folds, 1-h embargo (deployment scenario).

Scalers and all pre-processing steps are refitted inside each training set (scikit-learn pipelines).

## Environment

Python 3.13 with the versions pinned in `requirements.txt` (pandas 2.3.2, numpy 2.3.3, scikit-learn 1.7.2,
TPOT 1.1.0, matplotlib 3.10.8, dask 2026.3.0, openpyxl 3.1.5):

```
pip install -r requirements.txt
python scripts/01_construir_dataset.py
```

Paths are resolved relative to the repository root (`scripts/config.py`), so the repository can be cloned and run
from any location.

## Licence

Code: MIT License. Data (`dados/`) and results (`resultados/`): Creative Commons Attribution 4.0 (CC BY 4.0).
See `LICENSE`.

## How to cite

Rocha, L. A. G., dos Santos Safre, A. L., de Sousa, R. L., Guerrini, I. A. (2026). When is AutoML worth it? Data and
code [dataset and software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX (replace with the DOI minted by Zenodo for
the v1.0 release). See also `CITATION.cff`.

Corresponding author: Anderson Luiz dos Santos Safre, University of California Agriculture and Natural Resources
(andersonsafre@ucanr.edu).
