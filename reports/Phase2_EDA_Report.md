# Phase 2 — SDSS Exploratory Data Analysis Report

**Project:** Astronomical Object Classification Using Machine Learning  
**Data:** `sdss_cleaned.csv` | **Observations:** 99,999 | **Columns:** 18

## Dataset and target distribution

- GALAXY: 59,445 (59.45%)
- STAR: 21,593 (21.59%)
- QSO: 18,961 (18.96%)
- Majority-class baseline accuracy: 59.45%; future evaluations require class-wise recall and macro F1.
- Quality check: 0 null cells, 0 invalid photometric sentinels after Phase 1.

## Key quantitative findings

1. Five-band photometry is associated with the spectroscopic class labels, but class distributions overlap.
2. Adjacent `i` and `z` bands are strongly correlated: Pearson r = 0.9715. Correlation does not justify automatic feature deletion.
3. Median `u-g`: STAR = 1.319, GALAXY = 1.640, QSO = 0.393.
4. Median `g-r`: STAR = 0.527, GALAXY = 1.404, QSO = 0.214.
5. Highest single-feature eta-squared: `g_r` (0.326). This is descriptive class separation, **not** predictive accuracy.
6. Median redshifts STAR = -0.000, GALAXY = 0.456, QSO = 1.617. **Redshift is excluded from the primary photometry-only model.**

## Interpretation

Magnitudes are measurements on an inverse brightness scale: smaller magnitude generally means brighter observed flux. Colour indices are the difference between two magnitudes and summarize their relative spectral profile. The observed group differences suggest that both raw magnitudes and engineered colour features deserve evaluation in Phase 4/5.

The IQR rule is a diagnostic, not a deletion policy. The distributions and survey footprint reflect the observed sample, not a population-universal property of all celestial objects. Tiny statistical p-values in a very large dataset are not automatically evidence of practical predictive value.

## Next research questions

1. Do the four colour indices improve cross-validated macro F1 relative to the five-band baseline?
2. Which spectroscopic classes are confused in a purely photometric classifier?
3. How much performance changes if additional *spectroscopic* redshift is supplied, clearly labeled as a separate task?
4. Are results stable across stratified validation folds?

## Deliverables

- `notebooks/02_exploratory_data_analysis.ipynb`
- `figures/eda/*.png` (12 figures)
- `reports/phase2_*.csv`, `reports/phase2_validation_summary.json`
- `dashboard/dashboard_photometry_sample.csv` (balanced visualization sample)

## Sources and caveats

- https://www.sdss4.org/dr12/imaging/understanding-the-imaging-data/
- https://www.sdss4.org/dr17/algorithms/magnitudes/
- https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

SDSS magnitudes may be expressed as asinh magnitudes. The CSV alone does not specify a unique photometric measurement estimator or provide per-band measurement errors. Avoid claiming universal astrophysical rules from this selected catalog sample.
