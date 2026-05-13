# Mapping the Invisible — NeurIPS 2026 Main Track Submission

This directory contains the LaTeX source, figure scripts, and experimental pipeline for the paper.

## Layout

```
paper/
├── main.tex                    # Full paper (9 pages + appendix)
├── neurips_2026.sty            # Local fallback style file -- REPLACE before submission
├── checklist.tex               # Filled-in NeurIPS paper checklist
├── references.bib              # BibTeX bibliography
├── README.md                   # This file
├── figures/
│   ├── _style.py               # Shared matplotlib publication style
│   ├── 00_run_experiments.py   # Full 4-model 240-fold LOSO-CV (~45 min)
│   ├── 00_fast_results.py      # Fast variant: XGBoost-only LOSO (~15 min)
│   ├── 01_study_area.py        # Fig 1: study area + sensor distribution
│   ├── 02_model_comparison.py  # Fig 2: Random-split vs LOSO comparison
│   ├── 03_feature_importance.py# Fig 3: LightGBM feature importance
│   ├── 04_obs_vs_pred.py       # Fig 4: Observed vs predicted scatter
│   ├── 05_ej_analysis.py       # Fig 5: EJ correlation + quartile exceedance
│   ├── 06_fairness_error.py    # Fig 6: Error stratified by EJ quartile
│   ├── 07_spatial_deployment.py# Fig 7: Per-city mean-PM2.5 spatial fields
│   ├── results.json            # (generated) All experimental outputs
│   └── fig*.pdf                # (generated) Publication-quality PDF figures
```

## Before submission

1. **Replace `neurips_2026.sty`.** The included file is a local fallback that approximates the NeurIPS 2026 format. Download the official `neurips_2026.sty` from [https://neurips.cc/Conferences/2026](https://neurips.cc/Conferences/2026) (Paper Submission → Styles) and drop it in this directory.
2. **Author anonymization.** The paper is already anonymous; do not add author info until the camera-ready stage.
3. **Double-check references.** Cross-check any citation you rely on heavily against its source before the hard submission deadline.

## Reproducing the paper

From the repository root:

```bash
# 0. Install dependencies (Python 3.10+)
pip install -r backend/requirements.txt
pip install matplotlib scipy

# 1. Run the experimental pipeline (either variant):
cd paper/figures
python 00_run_experiments.py    # Full 4-model LOSO, ~45 min
# OR (faster):
python 00_fast_results.py       # XGBoost-only LOSO, ~15 min

# 2. Render figures:
python 01_study_area.py
python 02_model_comparison.py
python 03_feature_importance.py
python 04_obs_vs_pred.py
python 05_ej_analysis.py
python 06_fairness_error.py
python 07_spatial_deployment.py

# 3. Compile the paper:
cd ..
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Data

All experiments use `p2_processed.xls` from the repository root: 61,224 sensor-day records from 240 PurpleAir low-cost sensors spanning 201 Texas census tracts, January 1 -- December 30, 2025.

Feature schema: meteorological (humidity, temperature, pressure), temporal (month, day-of-year, day-of-week), PM2.5 autoregressive (1-day lag, 7-day lag, 7-day rolling mean), EJScreen socio-demographic and pollution-burden indicators (8 columns), and spatial (latitude, longitude). Target is daily mean PM2.5 (µg/m³), EPA Barkjohn-corrected.

## Compute

All experiments run on a single 10-core Apple M-series CPU with 16 GB RAM. Full LOSO-CV (4 models × 240 folds) completes in approximately 45 min wall-clock. Figure generation is nearly instant.

## License

MIT, once de-anonymized. Data use subject to PurpleAir Terms of Service.
