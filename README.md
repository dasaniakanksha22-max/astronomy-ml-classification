# Astronomical Object Classification Using Machine Learning
### SDSS Stellar Classification Project (SDSS17)

**Course:** DS605: Fundamentals of Machine Learning  
**Dataset:** SDSS Stellar Classification Dataset (SDSS17) — 100,000 observations, 18 columns  
**Classes:** `GALAXY` (59.45%), `STAR` (21.59%), `QSO` (18.96%)  
**Champion Model:** Random Forest Classifier (**88.20% Accuracy, 0.8524 Macro F1, 0.9653 ROC-AUC**)

---

## Project Structure

```text
ML Project/astronomy-ml-classification/
├── data/
│   ├── raw/
│   │   └── star_classification.csv             # Original SDSS17 dataset (100k rows)
│   └── processed/
│       ├── sdss_cleaned.csv                    # Phase 1: Cleaned data (99,999 rows, -9999 dropped)
│       └── sdss_features.csv                   # Phase 4: Engineered features & PCA projections
├── notebooks/
│   ├── 01_data_inspection_and_cleaning.ipynb   # Phase 1: Data Ingestion & Quality Control
│   ├── 02_exploratory_data_analysis.ipynb      # Phase 2: Exploratory Data Analysis & Statistics
│   ├── 03_dashboard_data_preparation.ipynb     # Phase 3: Dashboard Aggregations & Looker Prep
│   ├── 04_feature_engineering.ipynb            # Phase 4: Photometric Colors, Curvature & PCA
│   └── 05_model_training_and_evaluation.ipynb  # Phase 5: Multi-Model Benchmark & Evaluation
├── models/
│   ├── best_rf_model.joblib                    # Champion Random Forest Pipeline
│   ├── best_hgb_model.joblib                   # Histogram-Based Gradient Boosting Pipeline
│   ├── best_voting_model.joblib                # Soft Voting Ensemble (RF + HGB + LR)
│   ├── best_dt_model.joblib                    # Decision Tree Pipeline
│   ├── best_knn_model.joblib                   # k-Nearest Neighbors Pipeline
│   ├── best_svm_model.joblib                   # Linear Support Vector Machine Pipeline
│   ├── best_lr_model.joblib                    # Multinomial Logistic Regression Pipeline
│   ├── redshift_rf_model.joblib                # Scientific Ablation Model (RF + Redshift)
│   └── model_metadata.json                     # Training metadata, features & parameters
├── figures/
│   ├── eda/                                    # Phase 2: 12 EDA diagnostic plots
│   ├── feature_engineering/                    # Phase 4: Correlation, Scree plot, PCA 2D, MI
│   └── evaluation/                             # Phase 5: Confusion matrices, ROC curves, ablation
├── reports/
│   ├── Final_Project_Report.md                 # Complete academic project report
│   ├── Chalk_Talk_Preparation_Guide.md         # Comprehensive viva / defense guide with math
│   ├── model_comparison_results.csv            # Quantitative metrics across all models
│   ├── phase1_data_quality_report.json         # Phase 1 validation summary
│   ├── Phase2_EDA_Report.md                    # Phase 2 statistical summary
│   ├── phase3_validation.csv                   # Phase 3 dashboard validation
│   └── phase4_feature_summary.csv              # Phase 4 feature importance rankings
├── app.py                                      # Interactive Streamlit Web Application
├── train_models.py                             # Standalone model training & evaluation script
├── feature_engineering.py                      # Standalone feature engineering script
├── requirements.txt                            # Project dependencies
└── README.md                                   # Project documentation
```

---

## Quick Start & Running the Project

### 1. Launch the Interactive Web Application (Streamlit)

From the project directory or root folder, run:

```powershell
streamlit run app.py
```
*(Or via python: `python -m streamlit run app.py`)*

The web app provides:
- **🔭 Live Object Classifier:** Input $u, g, r, i, z$ magnitudes with sliders, select celestial presets (Star, Galaxy, Quasar), view instant classifications with confidence gauges, and see where the query lies in a dynamic 2D colour-colour diagram!
- **📁 Batch CSV Classification:** Drag-and-drop any catalog CSV to classify thousands of objects in seconds and download predictions with confidence scores.
- **📊 Model Performance Benchmark:** Interactive comparison of Accuracy, Macro F1, Confusion Matrices, and ROC curves across all 7 evaluated models.
- **🌌 Survey Map & EDA:** Interactive spatial distribution ($\alpha$ vs $\delta$) and photometric distributions.
- **📚 Chalk Talk & Viva Reference:** Mathematical formulations and syllabus-aligned questions and answers embedded directly in the application.

---

### 2. Running the Jupyter Notebooks

Launch Jupyter Lab or Notebook:

```powershell
jupyter lab
```

Navigate to `notebooks/` and execute sequentially:
1. `01_data_inspection_and_cleaning.ipynb`
2. `02_exploratory_data_analysis.ipynb`
3. `03_dashboard_data_preparation.ipynb`
4. `04_feature_engineering.ipynb`
5. `05_model_training_and_evaluation.ipynb`

All notebooks are pre-executed with cell outputs and figures saved.

---

### 3. Retraining Models via Standalone CLI

To re-run feature engineering and model training from terminal:

```powershell
python feature_engineering.py
python train_models.py
```

---

## Model Benchmark Summary (Test Set: $N = 20,000$)

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | $F1_{\text{Galaxy}}$ | $F1_{\text{Star}}$ | $F1_{\text{QSO}}$ | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Champion)** | **88.20%** | **0.8524** | **0.8824** | **0.9291** | **0.8128** | **0.8152** | **0.9653** |
| **Gradient Boosting** | 87.74% | 0.8481 | 0.8784 | 0.9256 | 0.8077 | 0.8111 | 0.9647 |
| **Voting Ensemble** | 87.71% | 0.8475 | 0.8778 | 0.9251 | 0.8062 | 0.8111 | 0.9551 |
| **k-Nearest Neighbors** | 86.74% | 0.8299 | 0.8654 | 0.9216 | 0.7704 | 0.7977 | 0.9474 |
| **Decision Tree** | 85.80% | 0.8257 | 0.8591 | 0.9119 | 0.7719 | 0.7933 | 0.9166 |
| **Linear SVM** | 73.89% | 0.6691 | 0.7306 | 0.8344 | 0.4632 | 0.7098 | 0.8335 |
| **Logistic Regression** | 67.86% | 0.6515 | 0.6885 | 0.7537 | 0.4864 | 0.7143 | 0.8352 |
| *Ablation: RF + Spectroscopic Redshift* | *97.56%* | *0.9721* | *0.9757* | *0.9856* | *0.9631* | *0.9752* | *0.9972* |

---

## Project Documentation & Viva Preparation
- **Comprehensive Project Report:** [reports/Final_Project_Report.md](file:///c:/Users/Akanksha/OneDrive/Desktop/ML_PROJECT2/ML%20Project/astronomy-ml-classification/reports/Final_Project_Report.md)
- **Complete Chalk Talk Viva Guide:** [reports/Chalk_Talk_Preparation_Guide.md](file:///c:/Users/Akanksha/OneDrive/Desktop/ML_PROJECT2/ML%20Project/astronomy-ml-classification/reports/Chalk_Talk_Preparation_Guide.md)
