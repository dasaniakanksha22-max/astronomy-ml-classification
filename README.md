# Astronomical Object Classification Using Machine Learning

## Phase 1 — Data ingestion, cleaning and pipeline setup

This project classifies SDSS objects as `STAR`, `GALAXY`, or `QSO` using five photometric magnitudes (`u`, `g`, `r`, `i`, `z`). The uploaded SDSS17 CSV contains 100,000 records and 18 columns. One row has invalid `-9999` measurements in `u`, `g`, and `z`, and is removed. The Phase 1 clean CSV contains **99,999 rows and 18 columns**.

### Folder structure

```text
Astronomical-Object-Classification/
├── data/
│   ├── raw/star_classification.csv
│   └── processed/sdss_cleaned.csv
├── notebooks/01_data_inspection_and_cleaning.ipynb
├── reports/phase1_data_quality_report.json
├── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

### Run on Windows (PowerShell)

Open a terminal in the `Astronomical-Object-Classification` project folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab
```

If Windows blocks PowerShell script activation, you can run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m jupyter lab` directly without activating the environment.

Open `notebooks/01_data_inspection_and_cleaning.ipynb`, select the Python environment and choose **Kernel → Restart Kernel and Run All Cells**. The notebook supports launching Jupyter from the project root or the `notebooks` folder.

### Important data decisions

- Keep the original source CSV unchanged.
- Preserve `obj_ID` and `spec_obj_ID` as strings; the exported scientific-notation identifiers are not fully precise.
- Drop only the record containing `-9999` in photometric columns.
- Keep repeated sky positions and negative spectroscopic redshifts; do not blindly mark them invalid.
- Do **not** fit an imputer or scaler on the full dataset. In Phase 4, fit preprocessing inside cross-validation or a train-only model pipeline.
- The primary model uses only `u, g, r, i, z`. `redshift` is retained for a **separate** later experiment.

### Phase 1 deliverables

- `notebooks/01_data_inspection_and_cleaning.ipynb`
- `data/processed/sdss_cleaned.csv`
- `reports/phase1_data_quality_report.json`

### Data provenance

Source: SDSS Stellar Classification Dataset (SDSS17): https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

### Git quick start

```bash
git init
git add .
git commit -m "Phase 1: inspect and clean SDSS stellar classification data"
```

Only use these commands if you are creating a new repository; for an existing project, add the files there and commit normally. Check any course-specific dataset sharing requirements before publishing raw data.
